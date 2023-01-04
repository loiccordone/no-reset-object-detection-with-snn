
from os.path import join
import sys
import argparse

try:
    import comet_ml
except ImportError:
    print("Comet is not installed, Comet logger will not be available.")

import torch
from torch import nn
from torch.utils.data import DataLoader

from torch.ao.quantization.fuse_modules import _fuse_modules
from fuse_other_modules import fuse_bn_conv, fuse_pad_conv

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from datasets.od_datasets import ContinuousSparseDetectionDataset, SingleFilesDetectionDataset, continuous_collate_fn
from object_detection_module import DetectionLitModule
from object_detection_module_fused import DetectionFusedLitModule
from models.postprocess_utils import PostProcessModule

def main():
    parser = argparse.ArgumentParser(description='Continuous object detection on an event dataset')
    # Dataset
    parser.add_argument('-dataset', default='gen1', type=str, help='dataset used {GEN1}')
    parser.add_argument('-path', default='../PropheseeGEN1', type=str, help='path to dataset location')
    parser.add_argument('-prefix', default='all_asynchro', type=str, help='prefix for the precomputed datasets filename')
    parser.add_argument('-num_classes', default=2, type=int, help='number of classes')

    # Data
    parser.add_argument('-b', default=32, type=int, help='batch size')
    parser.add_argument('-sample_size', default=50000, type=int, help='duration of a sample in µs')
    parser.add_argument('-T', default=1, type=int, help='simulating time-steps')
    parser.add_argument('-tbin', default=1, type=int, help='number of micro time bins')
    parser.add_argument('-image_shape', default=(240,304), type=tuple, help='spatial resolution of events')
    parser.add_argument('-spatial_q', default=(1,1), type=tuple, help='spatial quantization of events')

    # Training
    parser.add_argument('-epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('-lr', default=5e-3, type=float, help='learning rate used')
    parser.add_argument('-wd', default=1e-2, type=float, help='weight decay used')
    parser.add_argument('-beta', default=1., type=float, help='beta used in the smooth l1 loss')
    parser.add_argument('-num_workers', default=0, type=int, help='number of workers for dataloaders')
    parser.add_argument('-loss_modes', default=['train', 'val'], type=int, help='modes where the loss will be computed')
    parser.add_argument('-ap_modes', default=['val', 'test'], type=int, help='modes where the mAP will be computed')
    parser.add_argument('-no_train', action='store_false', help='whether to train the model', dest='train')
    parser.add_argument('-no_val', action='store_false', help='whether to use a val dataset', dest='val')
    parser.add_argument('-test', action='store_true', help='whether to test the model')
    parser.add_argument('-device', default=0, type=int, help='device')
    parser.add_argument('-precision', default=32, type=int, help='whether to use AMP {16, \'bf16\', 32, 64}')
    parser.add_argument('-save_ckpt', action='store_true', help='saves checkpoints')
    parser.add_argument('-comet_api', default=None, type=str, help='api key for Comet Logger')
    
    # Backbone
    parser.add_argument('-backbone', default='vgg', type=str, help='model used {vgg, rescat}', dest='model')
    parser.add_argument('-cfg', default='32, 32, 32, 64s2, 64, 128s2, 128, 128s2', type=str, help='configuration of the layers of the backbone')
    parser.add_argument('-no_bn', action='store_false', help='don\'t use BatchNorm2d', dest='bn')
    parser.add_argument('-pretrained_backbone', default=None, type=str, help='path to pretrained backbone model')
    parser.add_argument('-pretrained', default=None, type=str, help='path to pretrained model')
    parser.add_argument('-extras', type=str, default='256, 128, 128', help='number of channels for extra layers after the backbone')
    parser.add_argument('-one_one_extras', action='store_true', help='whether to use 1x1 3x3 extra layers')
    parser.add_argument('-dsod', action='store_true', help='whether to use DSOD heads')

    # Priors
    parser.add_argument('-min_ratio', default=0.05, type=float, help='min ratio for priors\' box generation')
    parser.add_argument('-max_ratio', default=0.80, type=float, help='max ratio for priors\' box generation')
    parser.add_argument('-aspect_ratios', default=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], type=int, help='aspect ratios for priors\' box generation')

    # Loss parameters
    parser.add_argument('-box_coder_weights', default=[10.0, 10.0, 5.0, 5.0], type=float, nargs=4, help='weights for the BoxCoder class')
    parser.add_argument('-iou_threshold', default=0.50, type=float, help='intersection over union threshold for the SSDMatcher class')
    parser.add_argument('-score_thresh', default=0.01, type=float, help='score threshold used for postprocessing the detections')
    parser.add_argument('-nms_thresh', default=0.45, type=float, help='NMS threshold used for postprocessing the detections')
    parser.add_argument('-topk_candidates', default=200, type=int, help='number of best detections to keep before NMS')
    parser.add_argument('-detections_per_img', default=100, type=int, help='number of best detections to keep after NMS')

    args = parser.parse_args()
    if args.model == "vgg":
        args.cfg = [int(el) if el.isdigit() else el for el in args.cfg.replace(' ', '').split(",")]
    elif args.model == "rescat":
        args.cfg = [int(el) for el in args.cfg.replace(' ', '').split(",")]
        
    args.extras = [int(el) for el in args.extras.replace(' ', '').split(",")]
    print(args)

    if args.dataset == "gen1":
        dataset = SingleFilesDetectionDataset
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")
        
    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
#         ckpt_path = join("pretrained", join(args.model, args.pretrained))
        ckpt_path = args.pretrained
        module = DetectionFusedLitModule.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)
    else:
        module = DetectionFusedLitModule(args)
        
    # FUSING BN AND PAD
    module.eval()
    
    ## MODULES TO FUSE
#     # Print modules to fuse to generate manually modules_to_fuse
#     for name, m in module.named_modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
#             print(name)
            
    backbone_modules = [[f"backbone.model.features.{i}.norm", f"backbone.model.features.{i}.conv"] for i in range(8)]
    extras_modules = [[f"backbone.extras.{i}.0.bn_conv_neuron.1", f"backbone.extras.{i}.0.bn_conv_neuron.2"] for i in range(3)]
    classif_head_modules = [[f"head.classification_head.module_list.{i}.1", f"head.classification_head.module_list.{i}.2"] for i in range(6)]
    reg_head_modules = [[f"head.regression_head.module_list.{i}.1", f"head.regression_head.module_list.{i}.2"] for i in range(6)]
    modules_to_fuse = backbone_modules + extras_modules + classif_head_modules + reg_head_modules

    fuse_custom_config_dict = {
        "additional_fuser_method_mapping": {
            (torch.nn.BatchNorm1d, torch.nn.Conv1d): fuse_bn_conv,
            (torch.nn.BatchNorm2d, torch.nn.Conv2d): fuse_bn_conv,
        },
    }
    
    fused_bn_module = _fuse_modules(module, modules_to_fuse, is_qat=False, fuse_custom_config_dict=fuse_custom_config_dict).to(module.device)
    print(fused_bn_module)
    
    ## FUSE PAD CONV
    # Print modules to fuse to generate manually modules_to_fuse
    for name, m in fused_bn_module.named_modules():
        if isinstance(m, nn.ConstantPad2d) or isinstance(m, nn.Conv2d):
            print(name)
            
    backbone_modules = [[f"backbone.model.features.{i}.pad", f"backbone.model.features.{i}.norm"] for i in range(1,8)]
    extras_modules = [[f"backbone.extras.{i}.0.bn_conv_neuron.0", f"backbone.extras.{i}.0.bn_conv_neuron.1"] for i in range(3)]
    classif_head_modules = [[f"head.classification_head.module_list.{i}.0", f"head.classification_head.module_list.{i}.1"] for i in range(6)]
    reg_head_modules = [[f"head.regression_head.module_list.{i}.0", f"head.regression_head.module_list.{i}.1"] for i in range(6)]
    modules_to_fuse = backbone_modules + extras_modules + classif_head_modules + reg_head_modules
    
    fuse_custom_config_dict = {
        "additional_fuser_method_mapping": {
            (torch.nn.ConstantPad1d, torch.nn.Conv1d): fuse_pad_conv,
            (torch.nn.ConstantPad2d, torch.nn.Conv2d): fuse_pad_conv,
        },
    }
    fused_pad_module = _fuse_modules(fused_bn_module, modules_to_fuse, is_qat=False, fuse_custom_config_dict=fuse_custom_config_dict).to(module.device)
    print(fused_pad_module)
        
    test_dataset = dataset(args, mode="test", prefix=args.prefix)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=args.b, 
        collate_fn=continuous_collate_fn, 
        num_workers=args.num_workers
    )
    
    trainer = pl.Trainer(
        gpus=[args.device], max_epochs=1,
        limit_train_batches=1., limit_val_batches=1.,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
    )

#     trainer.test(module, test_dataloader)
    trainer.test(fused_pad_module, test_dataloader)

        
if __name__ == '__main__':
    main()
    

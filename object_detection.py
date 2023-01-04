from os.path import join
import sys
import argparse

try:
    import comet_ml
except ImportError:
    print("Comet is not installed, Comet logger will not be available.")

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger

from datasets.od_datasets import ContinuousSparseDetectionDataset, continuous_collate_fn
from object_detection_module import DetectionLitModule

# Usage
# python object_detection.py -device 1 -path '../shared/PropheseeGEN1' -b 8 -comet_api comet_api_key -prefix 128_asynchro

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
    parser.add_argument('-cfg', default='32, 64, 64, 64s2, 64, 128s2, 128, 128s2', type=str, help='configuration of the layers of the backbone')
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
#         args.cfg = [32, 64, 64, 128, 128]
        # args.cfg = [32, (32, 64), (64, 128), (128, 128)]
        
    args.extras = [int(el) for el in args.extras.replace(' ', '').split(",")]
    print(args)

    if args.dataset == "gen1":
        dataset = ContinuousSparseDetectionDataset
    else:
        sys.exit(f"{args.dataset} is not a supported dataset.")
        
    # LOAD PRETRAINED MODEL
    if args.pretrained is not None:
#         ckpt_path = join("pretrained", join(args.model, args.pretrained))
        ckpt_path = args.pretrained
        module = DetectionLitModule.load_from_checkpoint(checkpoint_path=ckpt_path, strict=False)
    else:
        module = DetectionLitModule(args)

    callbacks=[]
    if args.save_ckpt:
        ckpt_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=f"ckpt-cont-od-{args.dataset}-{args.model}/",
            filename=f"{args.dataset}" + "-{epoch:02d}-{train_loss_epoch:.4f}",
            save_top_k=3,
            mode='min',
        )
        callbacks.append(ckpt_callback)

    logger = None
    if args.comet_api:
        try:
            comet_logger = CometLogger(
                api_key=args.comet_api,
                project_name=f"cont-od-{args.dataset}-{args.model}/",
                save_dir="comet_logs",
                log_code=True,
            )
            comet_logger.experiment.log_asset("object_detection_module.py")
            comet_logger.experiment.log_asset("models/spiking_thin_vgg.py")
            comet_logger.experiment.log_asset("models/spiking_rescat.py")
            logger = comet_logger
        except ImportError:
            print("Comet is not installed, Comet logger will not be available.")
            

    trainer = pl.Trainer(
        gpus=[args.device], max_epochs=args.epochs,
        limit_train_batches=1., limit_val_batches=1.,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
    )

    val_dataloader = None
    if args.val:
        val_dataset = dataset(args, mode="test", prefix=args.prefix)   
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.b, 
            collate_fn=continuous_collate_fn, 
            num_workers=args.num_workers
        ) 
        
    if args.train:
        train_dataset = dataset(args, mode="train", prefix=args.prefix)
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.b, 
            collate_fn=continuous_collate_fn, 
            num_workers=args.num_workers, 
            shuffle=True
        )
        
        trainer.fit(module, train_dataloader, val_dataloader)
        
    if args.test:
        test_dataset = dataset(args, mode="test", prefix=args.prefix)
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.b, 
            collate_fn=continuous_collate_fn, 
            num_workers=args.num_workers
        )

        trainer.test(module, test_dataloader)

if __name__ == '__main__':
    main()
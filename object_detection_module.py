from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor
import torchvision
import torchvision.models.detection._utils as det_utils
import torchvision.ops.boxes as box_ops
import pytorch_lightning as pl

import spikingjelly
from spikingjelly.clock_driven import functional

from models.detection_backbone import DetectionBackbone
from models.SSD_utils import GridSizeDefaultBoxGenerator, SSDHead, DSODHead, filter_boxes
from models.postprocess_utils import PostProcessModule, reshape_head_ouputs_list
from prophesee_utils.metrics.coco_utils import coco_eval

# from thop import profile
# from thop import clever_format

class DetectionLitModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.automatic_optimization = False
        
        self.save_hyperparameters()
        self.args = args
        self.lr = args.lr
        self.loss_modes, self.ap_modes = args.loss_modes, args.ap_modes
        
        self.backbone = DetectionBackbone(args)
        self.anchor_generator = GridSizeDefaultBoxGenerator(
            args.aspect_ratios, args.min_ratio, args.max_ratio)
        
        out_channels = self.backbone.out_channels
        print(out_channels)
#         assert len(out_channels) == len(self.anchor_generator.aspect_ratios)

        num_anchors = self.anchor_generator.num_anchors_per_location()
        self.head = SSDHead(out_channels, num_anchors, args.num_classes)
        
        self.box_coder = det_utils.BoxCoder(weights=args.box_coder_weights)
        self.proposal_matcher = det_utils.SSDMatcher(args.iou_threshold)

        self.all_nnz, self.all_nnumel = 0, 0

    def forward(self, events):
        features = self.backbone(events)
        head_outputs_list = self.head(features)

        # #TEST
        # all_detections = self.postprocess(head_outputs_list)
        # return all_detections

        #TRAIN/VAL
        head_outputs = reshape_head_ouputs_list(head_outputs_list, self.args.num_classes)
        return features, head_outputs
    
    def on_train_epoch_start(self):
        self.train_detections, self.train_targets = [], []

    def on_validation_epoch_start(self):
        self.val_detections, self.val_targets = [], []
        
    def on_test_epoch_start(self):
        self.backbone.add_hooks()
        self.test_detections, self.test_targets = [], []
        
#         def count_sj_plif(model, x, y):
#             x = x[0]
#             nelements = x.numel()
#             model.total_ops += 1*torch.DoubleTensor([int(nelements)])

#         def ignore_bn(model, x, y):
#             pass

#         x = torch.randn((1,2,240,304)).to(self.device)
#         flops, params = profile(self, inputs=(x, ),
#                                custom_ops={spikingjelly.clock_driven.neuron.ParametricLIFNode: count_sj_plif,
#                                           nn.BatchNorm2d: ignore_bn})
#         flops, params = clever_format([flops, params], "%.3f")
#         print(flops, params)
    
    def step(self, batch, batch_idx, mode):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        
        all_samples, all_targets = batch
        count_without_backward = 0

        for t, (events, targets) in enumerate(zip(all_samples, all_targets)):
            events = events.to(torch.float).to_dense().permute(0,1,-1,2,3).squeeze(1)
            features, head_outputs = self(events)

            # If test, measure activity
            if mode == "test":
                self.process_nz(self.backbone.get_nz_numel())
            
            idx_with_targets = [i for i,target in enumerate(targets) if target['boxes'].numel() > 0]
            nb_samples = len(idx_with_targets)

            if nb_samples > 0:
                # Keep only samples with targets
                features = [f[idx_with_targets] for f in features]
                head_outputs = {k: v[idx_with_targets] for k,v in head_outputs.items()}
                targets = [t for t in targets if t['boxes'].numel() > 0]
                
                # Anchors generation
                anchors = self.anchor_generator(features, self.args.image_shape)

                if mode in self.loss_modes:
                    # Match targets with anchors
                    matched_idxs = []
                    for anchors_per_image, targets_per_image in zip(anchors, targets):
                        # now we're sure that targets_per_image['boxes'].numel() > 0
                        match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                        matched_idxs.append(self.proposal_matcher(match_quality_matrix))

                    count_without_backward = 0
                    losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
                    # print(f"{t}. ({nb_samples}) Bbox_regression: {losses['bbox_regression'].item():.4f} / Classification: {losses['classification'].item():.4f}")

                    bbox_loss_t = losses['bbox_regression']
                    cls_loss_t = losses['classification']

                    self.log(f'{mode}_loss_bbox', bbox_loss_t, on_step=True, on_epoch=True, prog_bar=True)
                    self.log(f'{mode}_loss_classif', cls_loss_t, on_step=True, on_epoch=True, prog_bar=True)

                    loss_t = bbox_loss_t + cls_loss_t
                    self.log(f'{mode}_loss', loss_t, on_step=True, on_epoch=True, prog_bar=True)

                if mode == "train":
                    opt.zero_grad()
                    self.manual_backward(loss_t)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1.)
                    opt.step()
                    self.backbone.detach()
                    self.head.detach()
                    torch.cuda.empty_cache()

            else:
                count_without_backward += 1

            if mode in self.ap_modes and nb_samples > 0:
                detections = self.postprocess_detections(head_outputs, anchors)
                
                # Filter small boxes for test
                if mode != "train":
                    detections = list(map(filter_boxes, detections))
                    targets = list(map(filter_boxes, targets))
                    
                getattr(self, f"{mode}_detections").extend([{k: v.cpu().detach() for k,v in d.items()} for d in detections])
                getattr(self, f"{mode}_targets").extend([{k: v.cpu().detach() for k,v in t.items()} for t in targets])
                del detections

            # Detach if last backward was too many timesteps ago
            if count_without_backward > 10:
                count_without_backward = 0
                self.backbone.detach()
                self.head.detach()
                torch.cuda.empty_cache()
        
        functional.reset_net(self.backbone)
        torch.cuda.empty_cache()

        # Learning rate scheduler
        if mode == 'train':
            # step every epoch
            if self.trainer.is_last_batch:
                sch.step()
                
            # # step every 'n' batches
            # n_batches = 349 if self.args['nb'] == "all" else self.args['nb']
            # if mode == "train" and (batch_idx + 1) % (n_batches // 2) == 0:
            #     sch.step()
        
    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="train")
    
    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="val")
    
    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, mode="test")
    
    def on_mode_epoch_end(self, mode):
        print()
        if mode in self.ap_modes:
            print(f"[{self.current_epoch}] {mode} results:")
            
            targets = getattr(self, f"{mode}_targets")
            detections = getattr(self, f"{mode}_detections")

            if detections == []:
                print("No detections")
                return
            
            h, w = self.args.image_shape
            stats = coco_eval(
                targets, 
                detections, 
                height=h, width=w, 
                labelmap=("car", "pedestrian"))

            keys = [
                f'{mode}_AP_IoU=.5:.05:.95', f'{mode}_AP_IoU=.5', f'{mode}_AP_IoU=.75', 
                f'{mode}_AP_small', f'{mode}_AP_medium', f'{mode}_AP_large',
                f'{mode}_AR_det=1', f'{mode}_AR_det=10', f'{mode}_AR_det=100',
                f'{mode}_AR_small', f'{mode}_AR_medium', f'{mode}_AR_large',
            ]
            stats_dict = {k:v for k,v in zip(keys, stats)}
            self.log_dict(stats_dict)

        if mode == "test":
            print(f"Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
            self.all_nnz, self.all_nnumel = 0, 0
        
    def on_train_epoch_end(self):
        self.on_mode_epoch_end(mode="train")
        
    def on_validation_epoch_end(self):
        self.on_mode_epoch_end(mode="val")
        
    def on_test_epoch_end(self):
        self.on_mode_epoch_end(mode="test")
        
    def compute_loss(self, targets: List[Dict[str, Tensor]], 
                     head_outputs: Dict[str, Tensor], anchors: List[Tensor],
                     matched_idxs: List[Tensor]) -> Dict[str, Tensor]:
        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        num_foreground = 0
        bbox_loss, cls_loss = [], []
        
        # Match original targets with default boxes
        for (targets_per_image, 
             bbox_regression_per_image, 
             cls_logits_per_image, 
             anchors_per_image, 
             matched_idxs_per_image
             ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
            # produce the matching between boxes and targets
            foreground_idxs_per_image = matched_idxs_per_image >= 0
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground += foreground_idxs_per_image.sum()

            # Compute regression loss
            matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
            
            bbox_loss.append(
                nn.functional.smooth_l1_loss(
                    bbox_regression_per_image, 
                    target_regression, 
                    reduction="sum",
                    beta=self.args.beta,
                )
            )
            
            ## Compute classification loss (focal loss)
            gt_classes_target = torch.zeros_like(cls_logits_per_image)
            
            gt_classes_target[
                foreground_idxs_per_image,
                targets_per_image["labels"][foreground_matched_idxs_per_image],
            ] = 1.0
            
            cls_loss.append(
                torchvision.ops.focal_loss.sigmoid_focal_loss(
                    cls_logits_per_image,
                    gt_classes_target,
                    reduction="sum",
                )
            ) 

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)
        
        return {
            "bbox_regression": bbox_loss.sum() / max(1, num_foreground),
            "classification": cls_loss.sum() / max(1, num_foreground),
        }
    
    def postprocess_detections(
        self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor]
    ) -> List[Dict[str, Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_logits = head_outputs["cls_logits"]
                                             
        detections = []
        for boxes, logits, anchors in zip(bbox_regression, pred_logits, image_anchors):
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, self.args.image_shape)

            image_boxes, image_scores, image_labels = [], [], []
            for label in range(self.args.num_classes):
                logits_per_class = logits[:, label]
                score = torch.sigmoid(logits_per_class).flatten()
                
                # remove low scoring boxes
                keep_idxs = score > self.args.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                # keep only topk scoring predictions
                num_topk = min(self.args.topk_candidates, score.size(0))
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int32))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # non-maximum suppression
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.args.nms_thresh)
            keep = keep[: self.args.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections
        
    def process_nz(self, nz_numel):
        nz, numel = nz_numel
        total_nnz, total_nnumel = 0, 0

        for module, nnz in nz.items():
            if "act" in module:
                nnumel = numel[module]
                if nnumel != 0:
                    total_nnz += nnz
                    total_nnumel += nnumel
        if total_nnumel != 0:
            self.all_nnz += total_nnz
            self.all_nnumel += total_nnumel

    def configure_optimizers(self):
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Number of parameters:', n_parameters)
        
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr,
            weight_decay=self.args.wd,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            self.args.epochs,
            # eta_min=1e-5,
        )
        return [optimizer], [scheduler]

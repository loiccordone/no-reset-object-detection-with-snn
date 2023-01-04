import torch
from spikingjelly.clock_driven import functional

from models.SSD_utils import filter_boxes
from models.postprocess_utils import PostProcessModule
from object_detection_module import DetectionLitModule
from prophesee_utils.metrics.coco_utils import coco_eval

class DetectionFusedLitModule(DetectionLitModule):
    def __init__(self, args):
        super().__init__(args)

        self.postprocess = PostProcessModule(args)

    def forward(self, events):
        features = self.backbone(events)
        regression_heads, classification_heads = self.head(features)
        all_detections = self.postprocess(regression_heads, classification_heads)
        return all_detections
    
    def test_step(self, batch, batch_idx):
        all_samples, all_targets = batch

        for events, targets in zip(all_samples, all_targets):
            idx_with_targets = [i for i,target in enumerate(targets) if target['boxes'].numel() > 0]
            nb_samples = len(idx_with_targets)
            
            events = events.to(torch.float).to_dense().permute(0,1,-1,2,3).squeeze(1)
            all_detections = self(events)

            # Measure activity
            self.process_nz(self.backbone.get_nz_numel())

            # Contribute to mAP only if targets are present
            if nb_samples != 0:
                # Reconstruct dicts for mAP computation
                detections =  [{
                    "boxes": b,
                    "scores": s,
                    "labels": l,
                } for b,s,l in zip(*all_detections)]
                
                detections = list(map(filter_boxes, detections))
                targets = list(map(filter_boxes, targets))

                # No targets after filtering
                if any([t["boxes"].numel() == 0 for t in targets]):
                    continue
                    
                getattr(self, f"test_detections").extend([{k: v.cpu().detach() for k,v in d.items()} for d in detections])
                getattr(self, f"test_targets").extend([{k: v.cpu().detach() for k,v in t.items()} for t in targets])
                del detections
            
        functional.reset_net(self.backbone)

    def on_test_epoch_end(self):
        print()
        mode = "test"
        
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

        print(f"Total sparsity: {self.all_nnz} / {self.all_nnumel} ({100 * self.all_nnz / self.all_nnumel:.2f}%)")
        self.all_nnz, self.all_nnumel = 0, 0

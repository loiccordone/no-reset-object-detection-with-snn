import torch
import torch.nn as nn
import time
import torchvision
import math

def reshape_head_ouputs_list(regression_heads, classification_heads, num_classes=2):
    def reshape_head_outputs(head_outputs, num_columns):
        all_results = []
        for results in head_outputs:
            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N = results.shape[0]
            results = results.permute(0, 2, 3, 1)
            results = results.reshape(N, -1, num_columns)  # Size=(N, HWA, K)

            all_results.append(results)
        return torch.cat(all_results, dim=1)        
    return reshape_head_outputs(regression_heads, 4), reshape_head_outputs(classification_heads, num_classes)

class PostProcessModule(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes
        self.image_shape = args.image_shape
        
        self.score_thresh = args.score_thresh # remove all detections below (default: 0.01)
        self.iou_threshold = args.iou_threshold # remove all detections below (default: 0.5)
        self.nms_thresh = args.nms_thresh # non-maximum suppression (NMS) threshold (default: 0.45)
        self.topk_candidates = args.topk_candidates # number of best detections to keep before NMS (default: 200)
        self.detections_per_img = args.detections_per_img # number of best detections to keep after NMS (default: 100)
        self.box_coder_weights = args.box_coder_weights # BoxCoder weights (default: [10.0, 10.0, 5.0, 5.0])

        self.register_buffer("anchors", torch.load("anchors.pt"))
        
    def forward(self, regression_heads, classification_heads):
        # head_outputs_list is a tuple, each corresponding to "bbox_regression" and "cls_logits"
        regression_head_outputs, classification_head_outputs = reshape_head_ouputs_list(regression_heads, classification_heads)
                                             
        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, logits in zip(regression_head_outputs, classification_head_outputs):
            boxes = self.decode_single(boxes, self.anchors)
            boxes = self.clip_boxes_to_images(boxes)
            
            image_boxes, image_scores, image_labels = self.get_all_preds(boxes, logits)
            
            boxes, scores, labels = self.get_nms_preds(image_boxes, image_scores, image_labels)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return torch.stack(all_boxes), torch.stack(all_scores), torch.stack(all_labels)

    def decode_single(self, rel_boxes, boxes):
        # Code taken from torchvision.models.detection._utils.decode_single
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.
        """
        boxes = boxes.to(rel_boxes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.box_coder_weights
        dx = rel_boxes[:, 0::4] / wx
        dy = rel_boxes[:, 1::4] / wy
        dw = rel_boxes[:, 2::4] / ww
        dh = rel_boxes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max= math.log(1000.0 / 16))
        dh = torch.clamp(dh, max= math.log(1000.0 / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        # Distance from center to box's corner.
        c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
        c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

        pred_boxes1 = pred_ctr_x - c_to_c_w
        pred_boxes2 = pred_ctr_y - c_to_c_h
        pred_boxes3 = pred_ctr_x + c_to_c_w
        pred_boxes4 = pred_ctr_y + c_to_c_h
        pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
        return pred_boxes

    def clip_boxes_to_images(self, boxes):
        # Code taken from torchvision.ops.boxes.clip_boxes_to_images
        dim = boxes.dim()
        boxes_x = boxes[..., 0::2]
        boxes_y = boxes[..., 1::2]
        height, width = self.image_shape
        boxes_x = boxes_x.clamp(min=0, max=width)
        boxes_y = boxes_y.clamp(min=0, max=height)

        clipped_boxes = torch.stack((boxes_x, boxes_y), dim=dim)
        return clipped_boxes.reshape(boxes.shape)

    def get_all_preds(self, boxes, logits):
        image_boxes, image_scores, image_labels = [], [], []
        for label in range(self.num_classes):
            logits_per_class = logits[:, label]
            score = torch.sigmoid(logits_per_class).flatten()
            
            # remove low scoring boxes
            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]

            # keep only topk scoring predictions
            num_topk = min(self.topk_candidates, score.size(0))
            score, idxs = score.topk(num_topk)
            box = box[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int32))

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        
        return image_boxes, image_scores, image_labels

    def get_nms_preds(self, image_boxes, image_scores, image_labels):
        # keep_old = torchvision.ops.boxes.nms(image_boxes, image_scores, self.nms_thresh)
        keep = self.nms(image_boxes.to("cpu"), image_scores.to("cpu"), self.nms_thresh)

        # print(torch.allclose(keep_old, keep.to(keep_old)))
        
        keep = keep[: self.detections_per_img]

        return image_boxes[keep], image_scores[keep], image_labels[keep]

    def nms(self, boxes, scores, iou_threshold):
        """
        Adapted from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
        
        Performs non-maximum suppression (NMS) on boxes according
        to their intersection-over-union (IoU).

        NMS iteratively removes lower scoring boxes which have an
        IoU greater than iou_threshold with another (higher scoring) box.

        If multiple boxes have the exact same score and satisfy the IoU
        criterion with respect to a reference box, the selected box is
        not guaranteed to be the same between CPU and GPU. This is similar
        to the behavior of argsort in PyTorch when repeated values are present.

        Args:
            boxes (Tensor[N, 4])): boxes to perform NMS on. They
                are expected to be in ``(x1, y1, x2, y2)`` format with 
                ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
                WE EXPECT boxes TO BE ALREADY SORTED IN DESCENDING ORDER wrt scores
            scores (Tensor[N]): scores for each one of the boxes
                WE EXPECT scores TO BE ALREADY SORTED IN DESCENDING ORDER
            iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

        Returns:
            Tensor: int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
        """
        # t0 = time.time()
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        
        # 'order' keep the indices of bounding boxes with largest scores, in decreasing order
        # since 'boxes' and 'scores' are already sorted, it's just a range
        order = torch.arange(scores.shape[0])

        keep = []
        while len(order) > 0:
            # The index of largest confidence score
            idx = order[0]
            keep.append(idx)

            # compute ordinates for Intersection over Union (IoU)
            xx1 = x1[order[1:]].clamp(min=x1[idx].item())
            yy1 = y1[order[1:]].clamp(min=y1[idx].item())
            xx2 = x2[order[1:]].clamp(max=x2[idx].item())
            yy2 = y2[order[1:]].clamp(max=y2[idx].item())

            # Compute areas for IoU
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            intersection = w * h

            # Compute the IoU
            iou = intersection / (areas[idx] + areas[order[1:]] - intersection)
            ids = (iou<=iou_threshold).nonzero().flatten()
            if ids.numel() == 0:
                break
            order = order[ids+1]
    
        # print(f"Elapsed time: {time.time()-t0:.3f}s")

        return torch.tensor(keep, dtype=torch.int64)
        
        
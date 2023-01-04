from typing import Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from models.utils import SingleStepSpikingBlock
from spikingjelly.clock_driven import neuron

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def filter_boxes(tensors, min_box_diag=30, min_box_side=20):
    widths = tensors['boxes'][:,2] - tensors['boxes'][:,0] # get all widths
    heights = tensors['boxes'][:,3] - tensors['boxes'][:,1] # get all heights
    diag_square = widths**2 + heights**2
    mask = (diag_square >= min_box_diag**2) * (widths >= min_box_side) * (heights >= min_box_side)
    return {k:v[mask] for k,v in tensors.items()}

class GridSizeDefaultBoxGenerator(DefaultBoxGenerator):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, feature_maps, image_size):
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        n_images = feature_maps[0].shape[0]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        default_boxes = self._grid_default_boxes(grid_sizes, image_size, dtype=dtype)
        default_boxes = default_boxes.to(device)
        
        dboxes = []
        for _ in range(n_images):
            dboxes_in_image = default_boxes
            dboxes_in_image = torch.cat(
                [
                    dboxes_in_image[:, :2] - 0.5 * dboxes_in_image[:, 2:],
                    dboxes_in_image[:, :2] + 0.5 * dboxes_in_image[:, 2:],
                ],
                -1,
            )
            dboxes_in_image[:, 0::2] *= image_size[1]
            dboxes_in_image[:, 1::2] *= image_size[0]
            dboxes.append(dboxes_in_image)
        return dboxes


class SSDHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__()
        self.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return self.regression_head(x), self.classification_head(x)

    def detach(self):
        pass


class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def forward(self, x: List[Tensor]) -> Tensor:
        return tuple(self.module_list[i](features) for i, features in enumerate(x))

        '''
        all_results = []

        for i, features in enumerate(x):
            results = self.module_list[i](features)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)
        '''


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Sequential(
                nn.ConstantPad2d(1, 0.),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, num_classes * anchors, kernel_size=3, bias=False),
            ))
        cls_logits.apply(init_weights)
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Sequential(
                nn.ConstantPad2d(1, 0.),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, 4 * anchors, kernel_size=3, bias=False),
            ))
        bbox_reg.apply(init_weights)
        super().__init__(bbox_reg, 4)


class DSODHead(nn.Module):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        super().__init__()
        self.downsample_list = []

        for in_c, next_in_c in zip(in_channels[:-1], in_channels[1:]):
            self.downsample_list.append(SingleStepSpikingBlock(in_c, next_in_c, kernel_size=1, stride=2))
            # self.downsample_list.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))

        self.downsample_list = nn.ModuleList(self.downsample_list)
        self.downsample_list.apply(init_weights)
        
        print()
        print(in_channels)
        # in_channels = [in_channels[0]] + [in_channels[i] + in_channels[i-1] // 2 for i in range(1, len(in_channels))]
        in_channels = [in_channels[0]] + [2 * in_c for in_c in in_channels[1:]]
        print(in_channels)
        
        self.classification_head = DSODClassificationHead(in_channels, num_anchors, num_classes)
        self.regression_head = DSODRegressionHead(in_channels, num_anchors)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        last_features = x[0]
        cat_features = [last_features]
        for i, features in enumerate(x[1:]):
            last_features = self.downsample_list[i](last_features)
            cat_features.append(torch.cat([features, last_features], dim=1))
            last_features = features
        
        return {
            "bbox_regression": self.regression_head(cat_features),
            "cls_logits": self.classification_head(cat_features),
        }

    def detach(self):
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                if isinstance(m.v, torch.Tensor):
                    m.v.detach_()
                if isinstance(m.w, torch.Tensor):
                    m.w.detach_()

class DSODScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []
        
        for i, features in enumerate(x):
            results = self.module_list[i](features)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class DSODClassificationHead(DSODScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Sequential(
                nn.ConstantPad2d(1, 0.),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, num_classes * anchors, kernel_size=3, bias=False),
            ))
        cls_logits.apply(init_weights)
        super().__init__(cls_logits, num_classes)


class DSODRegressionHead(DSODScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Sequential(
                nn.ConstantPad2d(1, 0.),
                nn.BatchNorm2d(channels),
                nn.Conv2d(channels, 4 * anchors, kernel_size=3, bias=False),
            ))
        bbox_reg.apply(init_weights)
        super().__init__(bbox_reg, 4)
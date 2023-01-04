from collections import OrderedDict

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron as spj_neuron

class SpikingThinVGG(nn.Module):
    def __init__(self, num_init_channels, cfg, norm_layer=None, num_classes=1000, init_weights=True,
                 neuron: callable = None, **kwargs):
        super().__init__()
        
        self.activations = {}
        self.out_channels = []
        self.idx_pool = [i+1 for i, v in enumerate(cfg) if isinstance(v, str)]

        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)
        
        self.features = self.make_layers(num_init_channels, cfg=cfg,
                                         norm_layer=norm_layer, neuron=neuron, 
                                         bias=bias, **kwargs)

        if isinstance(cfg[-1], str):
            cfg[-1] = int(cfg[-1][:-2])
        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("norm_classif", norm_layer(cfg[-1])),
                    ("conv_classif", nn.Conv2d(cfg[-1], num_classes, 
                                                kernel_size=1, bias=bias)),
                ]
            )
        )
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x, classify=True):
        self.reset_activations()
        if classify:
            x = self.features(x)
            x = self.classifier(x)
            x = x.flatten(start_dim=-2).sum(dim=-1)
            return x
        else:
            fm_1 = x
            for i in range(self.idx_pool[0]):
                fm_1 = self.features[i](fm_1)
            fm_2 = fm_1
            for i in range(self.idx_pool[0], self.idx_pool[1]):
                fm_2 = self.features[i](fm_2)
            fm_3 = fm_2
            for i in range(self.idx_pool[1], self.idx_pool[2]):
                fm_3 = self.features[i](fm_3)
            return fm_1, fm_2, fm_3

    def make_layers(self, num_init_channels, cfg, norm_layer, neuron, bias, **kwargs):
        layers = nn.Sequential()
        in_channels = num_init_channels
        
        # Stem, patchify input
        layers.append(
            nn.Sequential(
                OrderedDict(
                    [
                        ("norm", norm_layer(in_channels)),
                        ("conv", nn.Conv2d(in_channels, cfg[0], kernel_size=4, padding=0, stride=4, bias=bias)),
                        # ("norm", norm_layer(v)),
                        ("act", neuron(**kwargs)),
                    ]
                )
            )
        )
        
        in_channels = cfg[0]
        stride = 1
        
        for v in cfg[1:]:
            if isinstance(v, str) and v.endswith('s2'):
                v = int(v[:-2])
                stride = 2

            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=0, stride=stride, bias=bias)
            layers.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            ("pad", nn.ConstantPad2d(1, 0.)),
                            ("norm", norm_layer(in_channels)),
                            ("conv", conv2d),
                            # ("norm", norm_layer(v)),
                            ("act", neuron(**kwargs)),
                        ]
                    )
                )
            )
            if stride == 2:
                self.out_channels.append(v)
                stride = 1
                    
            in_channels = v
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def add_hooks(self):
        def activation_at_module(name):
            def activation_hook(module, input, out):
                self.activations[name] = out
            return activation_hook
        
        self.hooks = {}
        for name, module in self.named_modules():
            self.hooks[name] = module.register_forward_hook(activation_at_module(name))
                
    def reset_activations(self):
        self.activations = {}
        
    def get_activations(self):
        return self.activations

def sequential_forward(sequential, x_seq):
    assert isinstance(sequential, nn.Sequential)
    out = x_seq
    for i in range(len(sequential)):
        m = sequential[i]
        if isinstance(m, spj_neuron.BaseNode):
            out = m(out)
        else:
            out = functional.seq_to_ann_forward(out, m)
    return out


class MultiStepSpikingThinVGG(SpikingThinVGG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, classify=True):
        self.reset_activations()
        if classify:
            x = sequential_forward(self.features, x)
            x = sequential_forward(self.classifier, x)
            x = x.flatten(start_dim=-2).sum(dim=-1)
            return x
        else:
            fm_1 = sequential_forward(self.features[:self.idx_pool[0]], x)
            fm_2 = sequential_forward(self.features[self.idx_pool[0]:self.idx_pool[1]], fm_1)
            fm_3 = sequential_forward(self.features[self.idx_pool[1]:], fm_2)
            return fm_1, fm_2, fm_3
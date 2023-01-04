from collections import OrderedDict

import torch
import torch.nn as nn
from spikingjelly.clock_driven import functional, neuron as spj_neuron

class TwoLayersResCatBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer: callable, bias: bool, neuron: callable, stride=1, **kwargs):
        super().__init__()
        
        self.pad1 = nn.ConstantPad2d(1, 0.)
        self.norm1 = norm_layer(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=0, bias=bias)
        self.act1 = neuron(**kwargs)
        
        self.pad2 = nn.ConstantPad2d(1, 0.)
        self.norm2 = norm_layer(inplanes)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=0, bias=bias)
        self.act2 = neuron(**kwargs)

        self.norm_trans = norm_layer(inplanes)
        self.conv_trans = nn.Conv2d(inplanes, planes//2, kernel_size=1, stride=stride, padding=0, bias=bias)
        self.act_trans = neuron(**kwargs)
        
#         self.trans = nn.MaxPool2d(2, 2, ceil_mode=True)

    def forward(self, x): 
        identity = self.act_trans(self.conv_trans(self.norm_trans(x)))
#         identity = self.trans(x)
        out = self.act1(self.conv1(self.norm1(self.pad1(x))))
        out = self.act2(self.conv2(self.norm2(self.pad2(out))))
        
        out = torch.cat([out, identity], 1)
        return out

class ResCatBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer: callable, bias: bool, neuron: callable, stride=1, **kwargs):
        super().__init__()
        
        self.pad1 = nn.ConstantPad2d(1, 0.)
        self.norm1 = norm_layer(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=0, bias=bias)
        self.act1 = neuron(**kwargs)

    def forward(self, x): 
        identity = x
        out = self.act1(self.conv1(self.norm1(self.pad1(x))))
        
        out = torch.cat([out, identity], 1)
        return out

class TransBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer: callable, bias: bool, neuron: callable, stride=2, **kwargs):
        super().__init__()
        
        self.norm = norm_layer(inplanes)
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=bias)
        self.act = neuron(**kwargs)

    def forward(self, x): 
        out = self.act(self.conv(self.norm(x)))
        return out

class SpikingResCat(nn.Module):
    def __init__(self, num_init_channels, cfg, norm_layer=None, num_classes=1000, init_weights=True,
                 neuron: callable = None, **kwargs):
        super().__init__()
        
        self.activations = {}
        self.out_channels = []

        if norm_layer is None:
            norm_layer = nn.Identity
        bias = isinstance(norm_layer, nn.Identity)

        stem = nn.Sequential(
            norm_layer(num_init_channels),
            nn.Conv2d(num_init_channels, cfg[0], kernel_size=4, stride=4, padding=0, bias=bias),
            neuron(**kwargs),
            nn.Sequential(OrderedDict([
                    ("pad", nn.ConstantPad2d(1, 0.)),
                    ("norm", norm_layer(cfg[0])),
                    ("conv", nn.Conv2d(cfg[0], cfg[1], kernel_size=3, padding=0, stride=1, bias=bias)),
                    ("act", neuron(**kwargs)),
            ])),
        )

        [32, 64, '64r', '64r', '128r']

        self.features = nn.Sequential(stem)
        # for in_channels, out_channels in cfg[2:]:
        last_channels = cfg[1]
        for in_channels in cfg[2:]:
            self.features.append(
                nn.Sequential(
                    # ResCatBlock(last_channels, in_channels // 2, norm_layer, bias=bias, neuron=neuron, stride=1, **kwargs),
                    # nn.Sequential(OrderedDict([
                    #         ("pad", nn.ConstantPad2d(1, 0.)),
                    #         ("norm", norm_layer(last_channels + in_channels // 2)),
                    #         ("conv", nn.Conv2d(last_channels + in_channels // 2, in_channels, kernel_size=3, padding=0, stride=2, bias=bias)),
                    #         ("act", neuron(**kwargs)),
                    # ])),
                    TwoLayersResCatBlock(last_channels, in_channels, norm_layer, bias=bias, neuron=neuron, stride=2, **kwargs),
                    # TransBlock(in_channels*3, out_channels, norm_layer, bias=bias, neuron=neuron, stride=2, **kwargs),
                )
            )
#             self.out_channels.append(in_channels + in_channels)
#             last_channels += in_channels
            self.out_channels.append(in_channels + in_channels // 2)
            last_channels = in_channels + in_channels // 2

        self.classifier = nn.Sequential(
            OrderedDict(
                [
                    ("norm_classif", norm_layer(self.out_channels[-1])),
                    ("conv_classif", nn.Conv2d(self.out_channels[-1], num_classes, 
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
            fm_1 = self.features[:2](x)
            fm_2 = self.features[2](fm_1)
            fm_3 = self.features[3](fm_2)
            return fm_1, fm_2, fm_3

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


class MultiStepSpikingResCat(SpikingResCat):
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
            fm_1 = sequential_forward(self.features[:2], x)
            fm_2 = sequential_forward(self.features[2], fm_1)
            fm_3 = sequential_forward(self.features[3], fm_2)
            return fm_1, fm_2, fm_3
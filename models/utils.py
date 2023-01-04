import torch.nn as nn

from spikingjelly.clock_driven import layer, neuron, surrogate
from .spiking_thin_vgg import SpikingThinVGG
from .spiking_rescat import SpikingResCat

class SingleStepSpikingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                   bias=False, stride=1, padding=0, v_reset=0.0, groups=1):
        super().__init__()
        self.bn_conv_neuron = nn.Sequential(
            nn.ConstantPad2d(padding, 0.),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=0, bias=bias, groups=groups),
            neuron.ParametricLIFNode(
                init_tau=2.0, v_threshold=1., 
                detach_reset=True, 
                v_reset=v_reset,
            ),
        )
        
    def forward(self, x):
        return self.bn_conv_neuron(x)


def get_model(args):
    norm_layer = nn.BatchNorm2d if args.bn else None

    if args.model == "vgg":
        return SpikingThinVGG(
            2*args.tbin, cfg=args.cfg, num_classes=2,
            norm_layer=norm_layer, neuron=neuron.ParametricLIFNode,
            detach_reset=True,
        )
    elif args.model == "rescat":
        return SpikingResCat(
            2*args.tbin, cfg=args.cfg, num_classes=2,
            norm_layer=norm_layer, neuron=neuron.ParametricLIFNode,
            detach_reset=True,
        )
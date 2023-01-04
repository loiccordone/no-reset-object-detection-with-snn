import torch
from torch import nn
from collections import OrderedDict

from models.utils import SingleStepSpikingBlock, get_model
from models.SSD_utils import init_weights
from spikingjelly.clock_driven import neuron

class DetectionBackbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.nz, self.numel = {}, {}

        # Necessary when loading pretrained weights, don't know why
        if not isinstance(args.cfg[-1], str) and 'vgg' in args.model:
            args.cfg = [c for c in args.cfg[:-1]] + [f"{args.cfg[-1]}s2"]
        self.model = get_model(args)
       
        if args.pretrained_backbone is not None:
            ckpt = torch.load(args.pretrained_backbone)
            state_dict = {k.replace('model.', ''):v for k,v in ckpt['state_dict'].items()}
            self.model.load_state_dict(state_dict, strict=False)
            
        self.out_channels = self.model.out_channels
        extras_fm = args.extras
        
        if args.one_one_extras:
            self.extras = nn.ModuleList(
                [
                    nn.Sequential(
                        SingleStepSpikingBlock(self.out_channels[-1], self.out_channels[-1]//2, kernel_size=1),
                        SingleStepSpikingBlock(self.out_channels[-1]//2, extras_fm[0], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SingleStepSpikingBlock(extras_fm[0], extras_fm[0]//2, kernel_size=1),
                        SingleStepSpikingBlock(extras_fm[0]//2, extras_fm[1], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        SingleStepSpikingBlock(extras_fm[1], extras_fm[1]//2, kernel_size=1),
                        SingleStepSpikingBlock(extras_fm[1]//2, extras_fm[2], kernel_size=3, padding=1, stride=2),
                    ),
                ]
            )
        else:
            n = neuron.ParametricLIFNode()
            self.extras = nn.ModuleList(
                [
                    nn.Sequential(
                        # SingleStepSpikingBlock(self.out_channels[-1], self.out_channels[-1], kernel_size=3, padding=1, stride=2, groups=self.out_channels[-1]),
                        # SingleStepSpikingBlock(self.out_channels[-1], extras_fm[0], kernel_size=1),
                        SingleStepSpikingBlock(self.out_channels[-1], extras_fm[0], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        # SingleStepSpikingBlock(extras_fm[0], extras_fm[0], kernel_size=3, padding=1, stride=2, groups=extras_fm[0]),
                        # SingleStepSpikingBlock(extras_fm[0], extras_fm[1], kernel_size=1),
                        SingleStepSpikingBlock(extras_fm[0], extras_fm[1], kernel_size=3, padding=1, stride=2),
                    ),
                    nn.Sequential(
                        # SingleStepSpikingBlock(extras_fm[1], extras_fm[1], kernel_size=3, padding=1, stride=2, groups=extras_fm[1]),
                        # SingleStepSpikingBlock(extras_fm[1], extras_fm[2], kernel_size=1),
                        SingleStepSpikingBlock(extras_fm[1], extras_fm[2], kernel_size=3, padding=1, stride=2),
                    ),
                ]
            )
            

        self.extras.apply(init_weights)
        self.out_channels.extend(extras_fm)
    
    def forward(self, x):
        self.reset_nz_numel()
        
        detection_feed = list(self.model(x, classify=False))
        x = detection_feed[-1]

        for block in self.extras:
            x = block(x)
            detection_feed.append(x)
            
        return detection_feed # [fm_1, fm_2, fm_3, fm_4, fm_5, fm_6]

    def detach(self):
        last_seq = None
        for m in self.modules():
            if isinstance(m, neuron.BaseNode):
                if isinstance(m.v, torch.Tensor):
                    m.v.detach_()
                if isinstance(m.w, torch.Tensor):
                    m.w.detach_()
            elif isinstance(m, nn.Sequential):
                last_seq = m

    def add_hooks(self):
        def get_nz(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[-1]
                if not isinstance(output, list):
                    self.nz[name] += torch.count_nonzero(output)
                    self.numel[name] += output.numel()
            return hook
        
        self.hooks = {}
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
            self.hooks[name] = module.register_forward_hook(get_nz(name))
                
    def reset_nz_numel(self):
        for name, module in self.named_modules():
            self.nz[name], self.numel[name] = 0, 0
        
    def get_nz_numel(self):
        return self.nz, self.numel
import torch
import torch.nn as nn
import copy

def fuse_pad_conv(is_qat, pad, conv):
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {pad},{conv}")
    else:
        return fuse_pad_conv_eval(pad, conv)

def fuse_pad_conv_eval(pad, conv):
    fused_conv = copy.deepcopy(conv)
    
    fused_conv.padding = (pad.padding[0], pad.padding[2])
    return fused_conv

def fuse_bn_conv(is_qat, bn, conv):
    r"""Given the bn and conv modules, fuses them and returns the fused module
    Args:
        is_qat: a flag for whether we are using quantization aware training fusion
        or post training quantization fusion
        bn: Spatial BN instance that needs to be fused with the conv
        conv: Module instance of type conv1d/conv2d
    Examples::
        >>> b1 = nn.BatchNorm2d(10)
        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> m2 = fuse_bn_conv(b1, m1)
    """
    
    if is_qat:
        raise NotImplementedError(f"Cannot fuse train modules: {bn},{conv}")
    else:
        if isinstance(conv, nn.Conv1d):
            return fuse_bn_conv_eval(bn, conv, fuse_bn_conv_weights_1d)
        elif isinstance(conv, nn.Conv2d):
            return fuse_bn_conv_eval(bn, conv, fuse_bn_conv_weights_2d)
        else:
            raise NotImplementedError(f"Cannot fuse {conv}. Fusion only supported for Conv1d and Conv2d.")
        
    
def fuse_bn_conv_eval(bn, conv, fuse_weights_nd):
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_weights_nd(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv


def fuse_bn_conv_weights_1d(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    print()
    print(conv_w.shape)
        
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    print((bn_w * bn_var_rsqrt).shape)
    print((bn_w * bn_var_rsqrt).view(1,-1,1).shape)
    new_conv_w = conv_w * (bn_w * bn_var_rsqrt).view(1,-1,1)
    print(new_conv_w.shape)
    
    bn_tmp = bn_b - (bn_rm * (bn_w * bn_var_rsqrt))
    bn_tmp_like_kernel = bn_tmp.view(1,-1,1).repeat(1, 1, conv_w.shape[2])
    print(bn_tmp_like_kernel.shape)
    new_conv_b = nn.functional.conv1d(bn_tmp_like_kernel, conv_w).squeeze()
    print(new_conv_b.shape)
    if conv_b is not None:
        new_conv_b = new_conv_b + conv_b
    
    return torch.nn.Parameter(new_conv_w), torch.nn.Parameter(new_conv_b)


def fuse_bn_conv_weights_2d(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
        
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
#     print(conv_w.shape)
#     print(bn_w.shape)
#     print((bn_w * bn_var_rsqrt).shape)
    new_conv_w = conv_w * (bn_w * bn_var_rsqrt).view(1,-1,1,1)
    
    bn_tmp = bn_b - (bn_rm * (bn_w * bn_var_rsqrt))
    bn_tmp_like_kernel = bn_tmp.view(1,-1,1,1).repeat(1, 1, conv_w.shape[2], conv_w.shape[3])
    new_conv_b = nn.functional.conv2d(bn_tmp_like_kernel, conv_w).squeeze()
    if conv_b is not None:
        new_conv_b = new_conv_b + conv_b

    return torch.nn.Parameter(new_conv_w), torch.nn.Parameter(new_conv_b)
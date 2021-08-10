# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/26/2021 9:03 AM
# @File:torch_fuse
import torch
import torchvision

"""
yolov5
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
        
def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
    # print('Fusing layers... ')
    for m in self.model.modules():
        if type(m) is Conv and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, 'bn')  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    # self.info()
    return self
"""

def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = torch.nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    x = torch.randn(16, 3, 256, 256)
    rn18 = torchvision.models.resnet18(pretrained=True)
    rn18.eval()
    net = torch.nn.Sequential(
        rn18.conv1,
        rn18.bn1
    )
    y1 = net.forward(x)
    fusedconv = fuse_conv_and_bn(net[0], net[1])
    y2 = fusedconv.forward(x)
    d = (y1 - y2).norm().div(y1.norm()).item()
    print("error: %.8f" % d)
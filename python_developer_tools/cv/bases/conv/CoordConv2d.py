
import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordConv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(CoordConv2d, self).__init__(
            in_chan + 2, out_chan, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        batchsize, H, W = x.size(0), x.size(2), x.size(3)
        h_range = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        w_range = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        h_chan, w_chan = torch.meshgrid(h_range, w_range)
        h_chan = h_chan.expand([batchsize, 1, -1, -1])
        w_chan = w_chan.expand([batchsize, 1, -1, -1])

        feat = torch.cat([h_chan, w_chan, x], dim=1)
        return F.conv2d(feat, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups)





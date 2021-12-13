# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/10/2021 2:41 PM
# @File:DepthwiseSeparableConvolution.py
import torch
from torch import nn

from python_developer_tools.cv.bases.channels.channels import ChannelShuffle


class SpatialSepConvSF(nn.Module):
    def __init__(self, inp, oups, kernel_size, stride):
        super(SpatialSepConvSF, self).__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup1,
                (kernel_size, 1),
                (stride, 1),
                (kernel_size//2, 0),
                bias=False, groups=1
            ),
            nn.BatchNorm2d(oup1),
            nn.Conv2d(oup1, oup1*oup2,
                (1, kernel_size),
                (1, stride),
                (0, kernel_size//2),
                bias=False, groups=oup1
            ),
            nn.BatchNorm2d(oup1*oup2),
            ChannelShuffle(oup1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class GroupConv(nn.Module):
    def __init__(self, inp, oup, groups=2):
        super(GroupConv, self).__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        print ('inp: %d, oup:%d, g:%d' %(inp, oup, self.groups[0]))
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False, groups=self.groups[0]),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class DepthConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False, groups=inp),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthSpatialSepConv(nn.Module):
    """深度空间可分离卷积"""
    def __init__(self, inp, expand, kernel_size, stride):
        super(DepthSpatialSepConv, self).__init__()

        exp1, exp2 = expand

        hidden_dim = inp * exp1
        oup = inp * exp1 * exp2

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * exp1,
                      (kernel_size, 1),
                      (stride, 1),
                      (kernel_size // 2, 0),
                      bias=False, groups=inp
                      ),
            nn.BatchNorm2d(inp * exp1),
            nn.Conv2d(hidden_dim, oup,
                      (1, kernel_size),
                      (1, stride),
                      (0, kernel_size // 2),
                      bias=False, groups=hidden_dim
                      ),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 3, 224, 224)
    dsconv = DepthwiseSeparableConvolution(3, 64)
    out = dsconv(input)
    print(out.shape)

    onnx_path = "netron_model.onnx"
    torch.onnx.export(dsconv, input, onnx_path, opset_version=11, verbose=True)
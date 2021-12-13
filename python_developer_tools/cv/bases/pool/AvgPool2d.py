# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/31/2021 1:37 PM
# @File:GlobalAvgPool2d
import torch.nn as nn

from python_developer_tools.cv.bases.activates.swish import h_swish


class GlobalAvgPool2d(nn.Module):
    """ Fast implementation of global average pooling from
        TResNet: High Performance GPU-Dedicated Architecture
        https://arxiv.org/pdf/2003.13630.pdf
    Args:
        flatten (bool, optional): whether spatial dimensions should be squeezed
    """
    def __init__(self, flatten: bool = False) -> None:
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)

class SwishAdaptiveAvgPool2d(nn.Module):
    def __init__(self,inplace=True):
        super().__init__()
        self.avgpool=nn.Sequential(
            nn.ReLU6(inplace=inplace),
            nn.AdaptiveAvgPool2d((1, 1)),
            h_swish()
        )
    def forward(self, x):
        return self.avgpool(x)
# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/31/2021 2:06 PM
# @File:spp
import torch.nn as nn
import torch

class SPP(nn.Module):
    """
        Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
        https://arxiv.org/pdf/1406.4729.pdf
    """
    def __init__(self, kernel_sizes):
        super().__init__()
        self.pools = [nn.MaxPool2d(k_size, stride=1, padding=k_size // 2) for k_size in kernel_sizes]

    def forward(self, x):
        feats = [x] + [pool(x) for pool in self.pools]
        return torch.cat(feats, dim=1)
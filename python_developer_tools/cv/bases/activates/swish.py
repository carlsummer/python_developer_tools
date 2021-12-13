# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:12/13/2021 10:08 AM
# @File:h_swish
import torch
import torch.nn as nn

from python_developer_tools.cv.bases.activates.sigmoid import h_sigmoid


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
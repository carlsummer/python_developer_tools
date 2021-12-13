# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:12/13/2021 3:01 PM
# @File:MaxGroupPooling
import torch.nn as nn
import torch

class MaxGroupPooling(nn.Module):
    def __init__(self, channel_per_group=2):
        super(MaxGroupPooling, self).__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x
        # max op
        b, c, h, w = x.size()

        # reshape
        y = x.view(b, c // self.channel_per_group, -1, h, w)
        out, _ = torch.max(y, dim=2)
        return out
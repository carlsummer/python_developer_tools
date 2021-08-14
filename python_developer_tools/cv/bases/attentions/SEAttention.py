# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/14/2021 3:05 PM
# @File:SEAttention.py
import numpy as np
import torch
from torch import nn
from torch.nn import init


class SEAttention(nn.Module):
    """
        Squeeze-and-Excitation Networks
        https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channel=512,reduction=16):
        super(SEAttention,self).__init__()
        mid_channel = channel // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SEAttention(channel=512,reduction=16)
    output=se(input)
    print(output.shape)
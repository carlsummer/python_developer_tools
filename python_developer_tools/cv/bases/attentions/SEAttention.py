# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/14/2021 3:05 PM
# @File:SEAttention.py
import numpy as np
import torch
from torch import nn
from torch.nn import init

from python_developer_tools.cv.bases.activates.sigmoid import h_sigmoid
from python_developer_tools.cv.bases.channels.channels import get_squeeze_channels


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


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.oup = oup
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # determine squeeze
        squeeze = get_squeeze_channels(inp, reduction)
        print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup),
            h_sigmoid()
        )

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup, 1, 1)
        return x_out * y

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SEAttention(channel=512,reduction=16)
    output=se(input)
    print(output.shape)
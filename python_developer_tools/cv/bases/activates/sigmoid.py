# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:12/13/2021 10:11 AM
# @File:sigmoid
import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

if __name__ == '__main__':
    a = torch.randn(4)
    a = torch.sigmoid(a)
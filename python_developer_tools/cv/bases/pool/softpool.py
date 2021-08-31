# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/11/2021 10:26 AM
# @File:softpool.py
import torch
import torch.nn as nn
try:
    import softpool_cuda
    from SoftPool import soft_pool2d, SoftPool2d
except ImportError:
    print('Please install SoftPool first: https://github.com/alexandrosstergiou/SoftPool')

# 使用softpool
# nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 可以用下面替换
# model_ft.maxpool = SoftPool2d(kernel_size=(2,2), stride=(2,2))

# 可以用的情况2
# self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
# self.pool2 = SoftPool2d(kernel_size=3, stride=2)

class SoftPool1D(torch.nn.Module):
    def __init__(self,kernel_size=2,stride=2):
        super(SoftPool1D, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(kernel_size,stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool


class SoftPool2D(nn.Module):
    """
        Refining activation downsampling with SoftPool
        PDF: https://arxiv.org/pdf/2101.00440v3.pdf
    """
    def __init__(self, kernel_size=2, stride=2):
        super(SoftPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class SoftPool3D(torch.nn.Module):
    def __init__(self,kernel_size,stride=2):
        super(SoftPool3D, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size,stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool

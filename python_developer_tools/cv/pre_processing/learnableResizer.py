# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:12/10/2021 3:29 PM
# @File:Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# > <br/>[paper]() <br/>
# > 尽管近年来卷积神经网络很大地促进了计算机视觉的发展，但一个重要方面很少被关注：图像大小对被训练的任务的准确性的影响。通常，输入图像的大小被调整到一个相对较小的空间分辨率(例如，224×224)，然后再进行训练和推理。这种调整大小的机制通常是固定的图像调整器（image resizer）（如：双行线插值）但是这些调整器是否限制了训练网络的任务性能呢？作者通过实验证明了典型的线性调整器可以被可学习的调整器取代，从而大大提高性能。虽然经典的调整器通常会具备更好的小图像感知质量（即对人类识别图片更加友好），本文提出的可学习调整器不一定会具备更好的视觉质量，但能够提高CV任务的性能。
# 在不同的任务重，可学习的图像调整器与baseline视觉模型进行联合训练。这种可学习的基于cnn的调整器创建了机器友好的视觉操作，因此在不同的视觉任务中表现出了更好的性能。作者使用ImageNet数据集来进行分类任务，实验中使用四种不同的baseline模型来学习不同的调整器，相比于baseline模型，使用本文提出的可学习调整器能够获得更高的性能提升。

class ResBlock(nn.Module):
    def __init__(self, num_channels=16):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.leakyrelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        return out

def make_block(r, n):
    residual = []

    for i in range(r):
        block = ResBlock(num_channels=n)
        residual.append(block)

    return nn.Sequential(*residual)

class ResizingNetwork(nn.Module):
    def __init__(self, r=1, n=16):
        super(ResizingNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n, kernel_size=7, stride=1, padding=3)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(n, n, kernel_size=1, stride=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(n)

        self.resblock = make_block(r, n)

        self.conv3 = nn.Conv2d(n, n, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

        self.conv4 = nn.Conv2d(n, out_channels=3, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        residual = F.interpolate(x, scale_factor=0.5, mode='bilinear')

        out = self.conv1(x)
        out = self.leakyrelu1(out)

        out = self.conv2(out)
        out = self.leakyrelu2(out)
        out = self.bn1(out)

        out_residual = F.interpolate(out, scale_factor=0.5, mode='bilinear')

        out = self.resblock(out_residual)

        out = self.conv3(out)
        out = self.bn2(out)
        out += out_residual

        out = self.conv4(out)
        out += residual

        return out
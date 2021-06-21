# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/18/2021 7:46 PM
# @File:sobel
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F


def nn_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(1, 1, 3, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel)
    # 对图像进行卷积操作
    edge_detect = conv_op(Variable(im))
    # 将输出转换为图片格式
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect


def functional_conv2d(im):
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    edge_detect = F.conv2d(Variable(im), weight)
    edge_detect = edge_detect.squeeze().detach().numpy()
    return edge_detect


def main():
    # 读入一张图片，并转换为灰度图
    im = Image.open('/home/deploy/datasets/creepageDistance/lr/org/6319938267001088_0_l..jpg').convert('L')
    # 将图片数据转换为矩阵
    im = np.array(im, dtype='float32')
    # 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
    im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
    # 边缘检测操作
    # edge_detect = nn_conv2d(im)
    edge_detect = functional_conv2d(im)
    # 将array数据转换为image
    im = Image.fromarray(edge_detect)
    # image数据转换为灰度模式
    im = im.convert('L')
    # 保存图片
    im.save('/home/deploy/workspaces/CreepageDistance/edge.jpg', quality=95)


if __name__ == "__main__":
    main()
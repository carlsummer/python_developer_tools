# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:12/13/2021 10:36 AM
# @File:DYMicroBlock
import torch.nn as nn
import torch
from yacs.config import CfgNode as CN

from python_developer_tools.cv.bases.activates.sigmoid import h_sigmoid
from python_developer_tools.cv.bases.attentions.SEAttention import SELayer
from python_developer_tools.cv.bases.channels.channels import _make_divisible, get_squeeze_channels, ChannelShuffle
from python_developer_tools.cv.bases.conv.DepthSpatialSepConvs import GroupConv, DepthSpatialSepConv, DepthConv


def get_pointwise_conv(mode, inp, oup, hiddendim, groups):
    if mode == 'group':
        return GroupConv(inp, oup, groups)
    elif mode == '1x1':
        return nn.Sequential(
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup)
                )
    else:
        return None

def get_act_layer(inp, oup, mode='SE1', act_relu=True, act_max=2, act_bias=True, init_a=[1.0, 0.0], reduction=4, init_b=[0.0, 0.0], g=None, act='relu', expansion=True):
    layer = None
    if mode == 'SE1':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction),
            nn.ReLU6(inplace=True) if act_relu else nn.Sequential()
        )
    elif mode == 'SE0':
        layer = nn.Sequential(
            SELayer(inp, oup, reduction=reduction),
        )
    elif mode == 'NA':
        layer = nn.ReLU6(inplace=True) if act_relu else nn.Sequential()
    elif mode == 'LeakyReLU':
        layer = nn.LeakyReLU(inplace=True) if act_relu else nn.Sequential()
    elif mode == 'RReLU':
        layer = nn.RReLU(inplace=True) if act_relu else nn.Sequential()
    elif mode == 'PReLU':
        layer = nn.PReLU() if act_relu else nn.Sequential()
    elif mode == 'DYShiftMax':
        layer = DYShiftMax(inp, oup, act_max=act_max, act_relu=act_relu, init_a=init_a, reduction=reduction, init_b=init_b, g=g, expansion=expansion)
    return layer


class DYShiftMax(nn.Module):
    def __init__(self, inp, oup, reduction=4, act_max=1.0, act_relu=True, init_a=[0.0, 0.0], init_b=[0.0, 0.0], relu_before_pool=False, g=None, expansion=False):
        super(DYShiftMax, self).__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(
                nn.ReLU(inplace=True) if relu_before_pool == True else nn.Sequential(),
                nn.AdaptiveAvgPool2d(1)
            )

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = _make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4
        print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        print('init-a: {}, init-b: {}'.format(init_a, init_b))

        self.fc = nn.Sequential(
                nn.Linear(inp, squeeze),
                nn.ReLU(inplace=True),
                nn.Linear(squeeze, oup*self.exp),
                h_sigmoid()
        )
        if g is None:
            g = 1
        self.g = g[1]
        if self.g !=1  and expansion:
            self.g = inp // self.g
        print('group shuffle: {}, divide group: {}'.format(self.g, expansion))
        self.gc = inp//self.g
        index=torch.Tensor(range(inp)).view(1,inp,1,1)
        index=index.view(1,self.g,self.gc,1,1)
        indexgs = torch.split(index, [1, self.g-1], dim=1)
        indexgs = torch.cat((indexgs[1], indexgs[0]), dim=1)
        indexs = torch.split(indexgs, [1, self.gc-1], dim=2)
        indexs = torch.cat((indexs[1], indexs[0]), dim=2)
        self.index = indexs.view(inp).type(torch.LongTensor)
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup*self.exp, 1, 1)
        y = (y-0.5) * self.act_max

        n2, c2, h2, w2 = x_out.size()
        x2 = x_out[:,self.index,:,:]

        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = torch.max(z1, z2)

        elif self.exp == 2:
            a1, b1 = torch.split(y, self.oup, dim=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out

class DYMicroBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, ch_exp=(2, 2), ch_per_group=4, groups_1x1=(1, 1), depthsep=True, shuffle=False, pointwise='fft', activation_cfg=None):
        super(DYMicroBlock, self).__init__()

        print(activation_cfg.dy)

        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = activation_cfg.dy
        act = activation_cfg.MODULE
        act_max = activation_cfg.ACT_MAX
        act_bias = activation_cfg.LINEARSE_BIAS
        act_reduction = activation_cfg.REDUCTION * activation_cfg.ratio
        init_a = activation_cfg.INIT_A
        init_b = activation_cfg.INIT_B
        init_ab3 = activation_cfg.INIT_A_BLOCK3

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                ChannelShuffle(hidden_dim2//2) if shuffle and y2 !=0 else nn.Sequential(),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)),
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle(oup//2) if shuffle and oup%2 == 0  and y3!=0 else nn.Sequential(),
            )
        elif g2 == 0:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = gs1,
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),

            )

        else:
            self.layers = nn.Sequential(
                get_pointwise_conv(pointwise, inp, hidden_dim2, hidden_dim1, gs1),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = False
                ) if y1 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(gs1[1]) if shuffle else nn.Sequential(),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride) if depthsep else
                DepthConv(hidden_dim2, hidden_dim2, kernel_size, stride),
                nn.Sequential(),
                get_act_layer(
                    hidden_dim2,
                    hidden_dim2,
                    mode=act,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    act_bias=act_bias,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g = gs1,
                    expansion = True
                ) if y2 > 0 else nn.ReLU6(inplace=True),
                ChannelShuffle(hidden_dim2//4) if shuffle and y1!=0 and y2 !=0 else nn.Sequential() if y1==0 and y2==0 else ChannelShuffle(hidden_dim2//2),
                get_pointwise_conv(pointwise, hidden_dim2, oup, hidden_fft, (g1, g2)), #FFTConv
                get_act_layer(
                    oup,
                    oup,
                    mode=act,
                    act_max=act_max,
                    act_relu=False,
                    act_bias=act_bias,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction//2 if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g = (g1, g2),
                    expansion = False
                ) if y3 > 0 else nn.Sequential(),
                ChannelShuffle(g2) if shuffle else nn.Sequential(),
                ChannelShuffle(oup//2) if shuffle and y3!=0 else nn.Sequential(),
            )

    def forward(self, x):
        identity = x
        out = self.layers(x)

        if self.identity:
            out = out + identity

        return out

if __name__ == '__main__':
    input_channel= 3
    stride = 2
    kernel_size =3
    activation_cfg = CN()
    activation_cfg.MODULE = "DYShiftMax" # 使用什么激活函数
    activation_cfg.ACT_MAX = 2.0 # 激活的倍数 激活函数的结果*2
    activation_cfg.LINEARSE_BIAS = False # act_bias
    activation_cfg.INIT_A_BLOCK3 = 1.0, 0.0
    activation_cfg.INIT_A = 1.0, 1.0
    activation_cfg.INIT_B = 0.0, 0.0
    activation_cfg.REDUCTION = 8
    activation_cfg.dy = [2, 0, 1]
    activation_cfg.ratio = 1
    output_channel = 8
    model =DYMicroBlock(input_channel, output_channel,
                kernel_size=kernel_size,
                stride=stride,
                ch_exp=(2, 2), # 深度空间可分离卷积等的通道expand
                ch_per_group=(0, 4), # 进行通道shuffle的分组数
                groups_1x1=(8, 2, 2), # hidden_fft(没有用到), g1(GroupConv 分组卷积的分组数), g2(没有用到)
                depthsep = True, # 是否对卷积进行深度可分离
                shuffle = True, # 是否shuffle通道
                pointwise = "group",
                activation_cfg=activation_cfg,
            )
    input = torch.randn(1024, 3, 32,32)
    out = model(input)
    print(out.shape)

    torch.onnx.export(model, input, "xx.onnx", input_names=['input'], output_names=['output'])
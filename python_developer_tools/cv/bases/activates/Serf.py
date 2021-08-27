# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/27/2021 11:06 AM
# @File:Serf
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Serf(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.erf(torch.log(1+torch.exp(x)))

def serf(x):
    """
    exp(n) 表示：e的n次方
    log(a) 表示：ln(a)
    # https://arxiv.org/pdf/2108.09598.pdf
    # f(x) = xerf(ln(1 + ex))
    """
    return x * torch.erf(torch.log(1+torch.exp(x)))


def convert_relu_to_serf(model):
    # 将relu替换为APReLU
    dyReluchannels = []
    for i, (m, name) in enumerate(zip(model.modules(), model.named_modules())):
        if type(m) is nn.ReLU:
            dyReluchannels.append({"name": name, "dyrelu": Serf()})
    for dictsss in dyReluchannels:
        setattr(model, dictsss["name"][0], dictsss["dyrelu"])
    return model

if __name__ == '__main__':
    x = torch.randn(3,64,64)
    print(serf(x))

    x = torch.linspace(-10, 10)
    y = serf(x)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.set_yticks([-1, -0.5, 0.5, 1])
    plt.plot(x, y, label='Softmax', linestyle='-', color='blue')
    plt.legend(['Softmax'])
    plt.show()
    # plt.savefig('softmax.png')
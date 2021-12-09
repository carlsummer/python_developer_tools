# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:11/19/2021 4:45 PM
# @File:GELU
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5*x*(1+F.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))


def gelu(x):
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*np.power(x,3))))

if __name__ == '__main__':
    # x = np.linspace(-4, 4, 10000)
    # y = gelu(x)
    x = torch.linspace(-4,4,10000)
    y = F.gelu(x)
    plt.plot(x, y)
    plt.show()
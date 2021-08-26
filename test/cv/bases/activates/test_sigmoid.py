# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/20/2021 8:55 AM
# @File:test_sigmoid
import torch
a = torch.randn(4)
a = torch.sigmoid(a)
print(a)

import torch
import torch.nn.functional as F
x= torch.Tensor( [ [1,2,3,4],[1,2,3,4],[1,2,3,4]])
y1= F.softmax(x, dim = 0) #对每一列进行softmax
print(y1)
# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 2:55 PM
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, _LRScheduler
import matplotlib.pyplot as plt

from python_developer_tools.cv.scheduler.poly_lr import PolyLR

initial_lr = 0.1
max_epoch = 100

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

net_1 = model()
optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
scheduler_1 = PolyLR(optimizer_1, max_iters=max_epoch)


print("初始化的学习率：", optimizer_1.defaults['lr'])

lr_list1 = []  # 把使用过的lr都保存下来，之后画出它的变化

for epoch in range(0, max_epoch):
    # train
    for iter in range(5):
        optimizer_1.zero_grad()
        optimizer_1.step()
        lr_list1.append(optimizer_1.param_groups[0]['lr'])
        scheduler_1.step(epoch)

# 画出lr的变化
fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
plt.plot(list(range(500)), lr_list1)
plt.xlabel("epoch")
plt.ylabel("lr")
plt.show()
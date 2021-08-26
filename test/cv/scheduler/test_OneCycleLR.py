# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/18/2021 9:48 PM
# @File:testsdf
import cv2
import torch.nn as nn
import torch
from torchvision.models import AlexNet
import matplotlib.pyplot as plt

# 定义2分类网络
steps = []
lrs = []
model = AlexNet(num_classes=2)
lr = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# total_steps:总的batch数,这个参数设置后就不用设置epochs和steps_per_epoch，anneal_strategy 默认是"cos"方式，当然也可以选择"linear"
# 注意这里的max_lr和你优化器中的lr并不是同一个
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.9, total_steps=100, verbose=True)
for epoch in range(10):
    for batch in range(10):
        scheduler.step()
        lrs.append(scheduler.get_lr()[0])
        steps.append(epoch * 10 + batch)

plt.figure()
plt.legend()
plt.plot(steps, lrs, label='OneCycle')
# plt.savefig("dd.png")
plt.show()
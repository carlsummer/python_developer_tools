# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/11/2021 10:26 AM
# @File:softpool.py
# 使用softpool
# nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 可以用下面替换
# model_ft.maxpool = SoftPool2d(kernel_size=(2,2), stride=(2,2))

# 可以用的情况2
# self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
# self.pool2 = SoftPool2d(kernel_size=3, stride=2)
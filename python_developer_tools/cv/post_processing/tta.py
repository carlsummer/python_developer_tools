# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/10/2021 1:37 PM
# @File:tta
import torch
import ttach
from ttach.base import Merger, Compose


def tta_Classification(x, model):
    transforms = Compose(
        [
            ttach.HorizontalFlip(),  # 水平翻转
            ttach.VerticalFlip(),  # 垂直翻转
        ]
    )
    tta_model = ttach.ClassificationTTAWrapper(model, transforms,merge_mode="mean")
    pre_batch = tta_model(x)
    return pre_batch

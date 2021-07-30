# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/30/2021 4:33 PM
# @File:wew
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/8 16:20
# @Author  : wangjianrong
# @File    : 1.模型串行.py
# 参考网址：https://blog.csdn.net/weixin_42264234/article/details/118633125
from torchvision.models.resnet import resnet50, resnet101
import random
import os
import numpy as np
import torch
from time import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED


def init_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Remove randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def inference(model, x, name):
    y = model(x)
    return name


def main():
    init_seed(0)
    s = time()
    fake_input = torch.randn(1, 3, 224, 224)
    e = time()
    print("gen data:", e - s)
    fake_input = fake_input.cuda()
    e = time()
    print("gen data:", e - s)
    warm_cnt = 100
    repeat = 100
    model1 = resnet50(True).cuda().eval()
    model2 = resnet101(True).cuda().eval()
    s = time()
    for i in range(warm_cnt):
        y = model1(fake_input)
    e = time()
    print("warm up res50:", e - s)
    s = time()
    for i in range(warm_cnt):
        y = model2(fake_input)
    e = time()
    print("warm up re101:", e - s)

    s = time()

    for i in range(repeat):
        y = inference(model1, fake_input, 1)
        y = inference(model2, fake_input, 1)

    e = time()
    print("模型串行耗时：", e - s)


if __name__ == '__main__':
    main()
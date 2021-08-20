# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/16/2021 4:02 PM
# @File:class_imgaug
import copy

import numpy as np
import cv2
import torch
from PIL import ImageDraw, ImageFont
from PIL import Image, ImageOps, ImageEnhance
import io
import random
from PIL import Image
import numpy as np


def img_autocontrast(img):
    """图片自动对比度"""
    edited = ImageOps.autocontrast(img, cutoff=3)
    return edited


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """直方图增强
       augment_hsv(img_hsv,0.0196,0.9,0.302)  # 进行颜色增强
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def mix_data(x, use_cuda=True, prob=0.6):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    p = random.uniform(0, 1)

    if p > prob:
        return x

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    fft_1 = torch.fft.fftn(x, dim=(1,2,3))
    abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

    fft_2 = torch.fft.fftn(x[index, :], dim=(1,2,3))
    abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

    fft_1 = abs_2*torch.exp((1j) * angle_1)

    mixed_x = torch.fft.ifftn(fft_1, dim=(1,2,3)).float()

    return mixed_x

if __name__ == '__main__':
    img_path = r'/home/zengxh/medias/data/ext/creepageDistance/lab_datasets/lr/org/6319938267001088_0_l..jpg'
    PIL_origin_image = Image.open(img_path)
    edited = img_autocontrast(PIL_origin_image)
    edited.save("hiseqpil_1.jpg")



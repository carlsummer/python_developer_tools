# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/16/2021 4:02 PM
# @File:class_imgaug
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageOps

def img_autocontrast():
    """图片自动对比度"""
    img_path = r'/home/zengxh/medias/data/ext/creepageDistance/lab_datasets/lr/org/6319938267001088_0_l..jpg'
    # img = cv2.imread(img_path)
    # cv2.imwrite("sdf.jpg",img)
    img = Image.open(img_path)  # name of the file is his_equi.jpg
    edited = ImageOps.autocontrast(img, cutoff=3)
    # edited.save("hiseqpil_1.jpg")
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
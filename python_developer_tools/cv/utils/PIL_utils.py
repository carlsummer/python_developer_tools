# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/30/2021 2:04 PM
# @File:PIL_utils
import cv2
import numpy as np
import numpy
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def PIL2cv2(image):
    """PIL转cv"""
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

def ImgText_CN(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    # 用于给图片添加中文字符
    if (isinstance(img, numpy.ndarray)):  # 判断是否为OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("font/simhei.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
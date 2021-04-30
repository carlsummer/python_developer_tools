# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 4:58 PM
# @File:image_utils
import cv2
import numpy as np
import base64
import re


def imgToBase64(imgpath):
    """将图片转base64"""
    with open(imgpath, 'rb') as f:
        base64_data = base64.b64encode(f.read())
        s = base64_data.decode()
    return s  # "data:image/jpeg;base64,{}".format(s)


def base64ToNparray(image_string):
    # 将base64图片转为numpy数组
    strinfo = re.compile(r'^data:image/\w+;base64,')
    image_string = strinfo.sub('', image_string)
    img_data = base64.b64decode(image_string)
    nparr = np.fromstring(img_data, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np


def opencvToBase64(imgCv):
    """opencv格式换位base64"""
    image = cv2.imencode('.jpg', imgCv)[1]
    base64_data = str(base64.b64encode(image))[2:-1]
    return base64_data

# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/30/2021 2:04 PM
# @File:PIL_utils
import cv2
import numpy as np


def PIL2cv2(image):
    """PIL转cv"""
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

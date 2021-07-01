# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/1/2021 2:32 PM
# @File:vis_lines
import numpy as np
import cv2

def line2mask(size, lines):
    H, W = size
    mask = np.zeros((H, W), np.uint8)
    for idx, l in enumerate(lines):
        x0 , y0, x1,y1  = l.coord
        cv2.line(mask, (x0, y0), (x1, y1), (idx+1), 1)
    return mask
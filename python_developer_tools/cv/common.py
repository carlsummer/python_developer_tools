# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 4:27 PM
# @File:common
import numpy as np


def computeIOU(pred_loc, ground_info):
    # 计算2个框的iou
    xmin1, ymin1, xmax1, ymax1 = [round(float(val)) for val in pred_loc]
    xmin2, ymin2, xmax2, ymax2 = [round(float(val)) for val in ground_info]

    # 求交集部分左上角的点
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    # 求交集部分右下角的点
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
    # 计算输入的两个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算总面积
    s = s1 + s2
    # 计算交集
    inter_area = (xmax - xmin) * (ymax - ymin)
    if s - inter_area == 0:
        return 1
    else:
        iou = inter_area / (s - inter_area)
        return round(iou, 2)


def bbox_expand(bbox, bbox_scale):
    # 对bbox进行缩放
    assert bbox_scale >= 0
    xmin, ymin, xmax, ymax = bbox
    h = ymax - ymin + 1
    new_h = h * bbox_scale
    w = xmax - xmin + 1
    new_w = w * bbox_scale
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    xmin = cx - 0.5 * new_w
    xmax = cx + 0.5 * new_w
    ymin = cy - 0.5 * new_h
    ymax = cy + 0.5 * new_h
    return np.array([xmin, ymin, xmax, ymax])

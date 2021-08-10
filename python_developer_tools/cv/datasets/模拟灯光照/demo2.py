# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/2/2021 12:48 PM
# @File:demo2
# 以图片为圆心，根据像素点与圆心的距离来进行不同程度的光照增强。
# coding:utf-8
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def stronglight(img,rows,cols,strength = 200):
    # rows h,cols w
    # strength设置光照强度

    # 设置中心点
    centerX = rows / 2
    centerY = cols / 2
    # print(centerX, centerY)
    radius = min(centerX, centerY)
    # print(radius)

    # 图像光照特效
    for i in range(rows):
        for j in range(cols):
            # 计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((centerY - j), 2) + math.pow((centerX - i), 2)
            # 获取原始图像
            B = img[i, j][0]
            G = img[i, j][1]
            R = img[i, j][2]
            if (distance < radius * radius):
                # 按照距离大小计算增强的光照值
                result = (int)(strength * (1.0 - math.sqrt(distance) / radius))
                B = img[i, j][0] + result
                G = img[i, j][1] + result
                R = img[i, j][2] + result
                # 判断边界 防止越界
                B = min(255, max(0, B))
                G = min(255, max(0, G))
                R = min(255, max(0, R))
                img[i, j] = np.uint8((B, G, R))
            else:
                img[i, j] = np.uint8((B, G, R))
    return img

if __name__ == '__main__':
    # 读取原始图像
    img = cv2.imread('2020063010140796.jpg')

    # 获取图像行和列
    rows, cols = img.shape[:2]

    img = stronglight(img, rows, cols)

    # 显示图像
    cv2.imwrite('test.jpg', img)
    plt.imshow(img)
    plt.show()

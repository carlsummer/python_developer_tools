# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/24/2021 8:59 AM
# @File:funcc2
import time

import cv2
import aircv as ac
import matplotlib.pyplot as plt

IMAGE_SIZE = 800


# 将图片缩放成短边IMAGE_SIZE
def changeImageSize(inputImage, x, y):
    xymin = min(x, y)
    rates = xymin / IMAGE_SIZE
    outputImage = cv2.resize(inputImage, (int(y / rates), int(x / rates)))
    return outputImage


# 判断短边是否小于IMAGE_SIZE，是的话执行resize
def judgeChangeable(image):
    h, w = image.shape[:2]
    if h > IMAGE_SIZE and w > IMAGE_SIZE:
        img = changeImageSize(image, h, w)
        return img
    else:
        return image


srcIm = ac.imread('../../../../../../../git-chint-workspace/help_cell/6500139267002166.jpg')
objIm = ac.imread('../../../../../../../git-chint-workspace/help_cell/6500139267002166_5_18.jpg')

start = time.time()
# objIm = judgeChangeable(objIm)
# print(objIm.shape)

match_res = ac.find_template(srcIm, objIm, 0.5)
print(match_res)

if match_res is not None:
    rect_points = match_res['rectangle']
    TL = rect_points[0]
    BR = rect_points[3]
    print(time.time()-start)
    # img2 = srcIm[TL[1]:BR[1], TL[0]:BR[0]]
    # cv2.rectangle(srcIm,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
    cv2.rectangle(srcIm, (TL[0], TL[1]), (BR[0], BR[1]), (255, 0, 0), 2)
    cv2.imwrite("../../../../../../../git-chint-workspace/help_cell/sdf.jpg", srcIm)
else:
    print("Can't find objIm site")
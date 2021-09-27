# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/14/2021 4:07 PM
# @File:BackgroundSubtractorMOG2.py
"""
41.3-BackgroundSubtractorMOG2.py:
这个算法的一个特点是它为每一个像素选择一个合适数目的 斯分布。
上一个方法中我们使用是 K 给斯分 布 。
 这样就会对由于亮度等发生变化引起的场景变化产生更好的适应。
和前面一样我们  创建一个背景对 。但在  我们我们可以 择是否 检测阴影。如果 detectShadows = True 默认值
它就会检测并将影子标记出来 但是 样做会降低处理速度。影子会 标记为灰色。
"""

import numpy as np
import cv2

# cap = cv2.VideoCapture('../data/vtest.avi')
cap = cv2.VideoCapture(0)#笔记本摄像头

fgbg = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, flipCode=1)  # 左右翻转

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) #& 0xff
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
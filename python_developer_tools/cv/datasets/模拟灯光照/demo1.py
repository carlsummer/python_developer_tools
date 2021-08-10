# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/2/2021 12:41 PM
# @File:光照增强
# 选取需要光照增强的区域，直接将像素值增加到240-255之间。
import cv2
import matplotlib.pyplot as plt
import random
img = cv2.imread('2020063010140796.jpg')
h,w,c = img.shape
print(h,w)
start_x = 510
start_y = 1000
for i in range(200):
    for j in range(300):
        if img[start_x+i,start_y+j][0] > 40:
            a = random.randint(250,255)
            b = random.randint(250,255)
            c = random.randint(250,255)
            img[start_x+i,start_y+j][0] = a
            img[start_x+i,start_y+j][1] = b
            img[start_x+i,start_y+j][2] = c
cv2.imwrite('test.jpg', img)
plt.imshow(img)
plt.show()
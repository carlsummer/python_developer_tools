# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/21/2021 10:05 PM
# @File:LSD
#https://github.com/primetang/LSD-OpenCV-MATLAB/tree/master/pylsd
#pip install ocrd-fork-pylsd==0.0.3
#pip install pylsd
import cv2
import numpy as np
import os
from pylsd.lsd import lsd
fullName = "/home/deploy/zengxiaohui/data/ext/creepageDistance/20210621/sunchangzhen/lr/org/750763926004462_1_l.jpg"
folder, imgName = os.path.split(fullName)
src = cv2.imread(fullName, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
lines = lsd(gray)
lines = np.concatenate((lines, np.abs(lines[:, 0] - lines[:, 2])[:, None], np.abs(lines[:, 1] - lines[:, 3])[:, None]),axis=1)
angle = np.arctan(lines[:, 6] / lines[:, 5])
lines = np.concatenate((lines, angle[:, None]), axis=1)  # x1,y1,x2,y2,line_width,width,height,angle
for i in range(lines.shape[0]):
    pt1 = (int(lines[i, 0]), int(lines[i, 1]))
    pt2 = (int(lines[i, 2]), int(lines[i, 3]))
    width = lines[i, 4]
    cv2.line(src, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))
cv2.imwrite('lsd.jpg', src)

# import cv2
# import math
# import numpy as np
#
# #Read gray image
# img = cv2.imread("/home/deploy/zengxiaohui/data/ext/creepageDistance/20210621/sunchangzhen/lr/org/750763926004462_1_l.jpg",0)
#
# #Create default parametrization LSD
# lsd = cv2.createLineSegmentDetector(0)
#
# #Detect lines in the image
# lines = lsd.detect(img)[0] #Position 0 of the returned tuple are the detected lines
# print(lines[0][0][0])
#
# ver_lines = []
#
# for line in lines:
#     angletan = math.degrees(math.atan2((round(line[0][3],2) - round(line[0][1],2)), (round(line[0][2],2) - round(line[0][0],2))))
#
#     if(angletan > 85 and angletan < 95):
#         ver_lines.append(line)
#
# #Draw detected lines in the image
# drawn_img = lsd.drawSegments(img,np.array(ver_lines))
#
# #Show image
# # cv2.imshow("LSD",drawn_img )
# # cv2.waitKey(0)
# cv2.imwrite("LSD.jpg",drawn_img )

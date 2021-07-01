# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/23/2021 1:16 PM
# @File:hough_utils
import numpy as np
from skimage.measure import label, regionprops
import cv2

class Line(object):
    def __init__(self, coordinates=[0, 0, 1, 1]):
        """
        coordinates: [y0, x0, y1, x1]
        """
        assert isinstance(coordinates, list)
        assert len(coordinates) == 4
        assert coordinates[0]!=coordinates[2] or coordinates[1]!=coordinates[3]
        self.__coordinates = coordinates

    @property
    def coord(self):
        return self.__coordinates

    @property
    def length(self):
        start = np.array(self.coord[:2])
        end = np.array(self.coord[2::])
        return np.sqrt(((start - end) ** 2).sum())

    def angle(self):
        y0, x0, y1, x1 = self.coord
        if x0 == x1:
            return -np.pi / 2
        return np.arctan((y0-y1) / (x0-x1))

    def rescale(self, rh, rw):
        coor = np.array(self.__coordinates)
        r = np.array([rh, rw, rh, rw])
        self.__coordinates = np.round(coor * r).astype(np.int).tolist()

    def __repr__(self):
        return str(self.coord)

def get_hough_space_label(lines,numangle,numrho,newH,newW):
    # 将线段转变为霍夫特征 lines = [line] = [[y0, x0, y1, x1]]
    hough_space_label = np.zeros((numangle, numrho))
    for l in lines:
        theta, r = line2hough(l, numAngle=numangle, numRho=numrho, size=(newH, newW))
        hough_space_label[theta, r] += 1

    hough_space_label = cv2.GaussianBlur(hough_space_label, (5, 5), 0)

    if hough_space_label.max() > 0:
        hough_space_label = hough_space_label / hough_space_label.max()
    return hough_space_label

def hough2points(hough_space_label,numangle,numrho,newH,newW):
    # 霍夫特征转成点
    kmap_label = label(hough_space_label, connectivity=1)
    props = regionprops(kmap_label)
    plist = []
    for prop in props:
        plist.append(prop.centroid)
    b_points = reverse_mapping(plist, numAngle=numangle, numRho=numrho, size=(newH,newW))
    return b_points

def get_boundary_point(y, x, angle, H, W):
    '''
    Given point y,x with angle, return a two point in image boundary with shape [H, W]
    return point:[x, y]
    '''
    point1 = None
    point2 = None

    if angle == -np.pi / 2:
        point1 = (x, 0)
        point2 = (x, H - 1)
    elif angle == 0.0:
        point1 = (0, y)
        point2 = (W - 1, y)
    else:
        k = np.tan(angle)
        if y - k * x >= 0 and y - k * x < H:  # left
            if point1 == None:
                point1 = (0, int(y - k * x))
            elif point2 == None:
                point2 = (0, int(y - k * x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if k * (W - 1) + y - k * x >= 0 and k * (W - 1) + y - k * x < H:  # right
            if point1 == None:
                point1 = (W - 1, int(k * (W - 1) + y - k * x))
            elif point2 == None:
                point2 = (W - 1, int(k * (W - 1) + y - k * x))
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x - y / k >= 0 and x - y / k < W:  # top
            if point1 == None:
                point1 = (int(x - y / k), 0)
            elif point2 == None:
                point2 = (int(x - y / k), 0)
                if point2 == point1: point2 = None
        # print(point1, point2)
        if x - y / k + (H - 1) / k >= 0 and x - y / k + (H - 1) / k < W:  # bottom
            if point1 == None:
                point1 = (int(x - y / k + (H - 1) / k), H - 1)
            elif point2 == None:
                point2 = (int(x - y / k + (H - 1) / k), H - 1)
                if point2 == point1: point2 = None
        # print(int(x-y/k+(H-1)/k), H-1)
        if point2 == None: point2 = point1
    return point1, point2

def reverse_mapping(point_list, numAngle, numRho, size=(32, 32)):
    #return type: [(y1, x1, y2, x2)]
    H, W = size
    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle
    b_points = []

    for (thetai, ri) in point_list:
        theta = thetai * itheta
        r = ri - numRho // 2
        cosi = np.cos(theta) / irho
        sini = np.sin(theta) / irho
        if sini == 0:
            x = np.round(r / cosi + W / 2)
            b_points.append((0, int(x), H-1, int(x)))
        else:
            # print('k = %.4f', - cosi / sini)
            # print('b = %.2f', np.round(r / sini + W * cosi / sini / 2 + H / 2))
            angle = np.arctan(- cosi / sini)
            y = np.round(r / sini + W * cosi / sini / 2 + H / 2)
            p1, p2 = get_boundary_point(int(y), 0, angle, H, W)
            if p1 is not None and p2 is not None:
                b_points.append((p1[1], p1[0], p2[1], p2[0]))
    return b_points

def convert_line_to_hough(line, size=(32, 32)):
    H, W = size
    theta = line.angle()
    alpha = theta + np.pi / 2
    if theta == -np.pi / 2:
        r = line.coord[1] - W/2
    else:
        k = np.tan(theta)
        y1 = line.coord[0] - H/2
        x1 = line.coord[1] - W/2
        r = (y1 - k*x1) / np.sqrt(1 + k**2)
    return alpha, r

def line2hough(line, numAngle, numRho, size=(32, 32)):
    # 直线转霍夫特征
    H, W = size
    alpha, r = convert_line_to_hough(line, size)

    irho = int(np.sqrt(H*H + W*W) + 1) / ((numRho - 1))
    itheta = np.pi / numAngle

    r = int(np.round(r / irho)) + int((numRho) / 2)
    alpha = int(np.round(alpha / itheta))
    if alpha >= numAngle:
        alpha = numAngle - 1
    return alpha, r

"""
import cv2
import numpy as np
# 1.加载图片，转为二值图
img = cv2.imread('shapes.jpg')
drawing = np.zeros(img.shape[:], dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
# 2.霍夫直线变换
lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 90)
参数1：要检测的二值图（一般是阈值分割或边缘检测后的图）
参数2：距离 ρ 的精度，值越大，考虑越多的线
参数3：角度 θ 的精度，值越小，考虑越多的线
参数4：累加数阈值，值越小，考虑越多的线
# 3.将检测的线画出来（注意是极坐标噢）
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(drawing, (x1, y1), (x2, y2), (0, 0, 255))
    
drawing = np.zeros(img.shape[:], dtype=np.uint8)
# 统计概率霍夫线变换
lines = cv2.HoughLinesP(edges, 0.8, np.pi / 180, 90, minLineLength=50, maxLineGap=10)
# 将检测的线画出来
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType = cv2.LINE_AA)
cv2.imwrite('houghlines4.jpg', drawing)

minLineLength：最短长度阈值，比这个长度短的线会被排除
maxLineGap：同一直线两点之间的最大距离
"""
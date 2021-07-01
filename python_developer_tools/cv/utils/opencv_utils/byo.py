# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/2/2021 12:56 PM
# @File:byo
import os

import copy
import cv2
import numpy as np
from ..opencv_utils.new_find import graytoprofile
from python_developer_tools.files.common import resetDir


def to_gray(image):
    # 将图片转灰度图
    if len(image.shape) == 2:
        pass
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise Exception(('Image shape error: {}').format(image.shape))
    return image


def compress_image(rgbImg_show2, ratio):
    # 压缩图片
    return cv2.resize(rgbImg_show2, (0, 0), fx=ratio, fy=ratio)


def opencvToBytes(frame):
    # opencv 转字节流
    return cv2.imencode(".jpg", frame)[1].tobytes()


def get_adaptive_threshold(dilation_subImg):
    # 自适应二值化
    return cv2.threshold(dilation_subImg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def get_threshold(image):
    threshold_binary = 90
    retval, threshold_img = cv2.threshold(image, threshold_binary, 255, cv2.THRESH_BINARY)
    return threshold_img


def get_scharr(o, ddepth=cv2.CV_64F):
    """获取scharr的特征"""
    scharrx = cv2.Scharr(o, ddepth, 1, 0)
    scharry = cv2.Scharr(o, ddepth, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)

    scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    return scharrx, scharry, scharrxy


def get_sobel(img, ddepth=cv2.CV_64F):
    x = cv2.Sobel(img, ddepth, 1, 0)
    y = cv2.Sobel(img, ddepth, 0, 1)

    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return absX, absY, dst


def get_laplace(gray):
    laplace = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    return laplace


def get_blur(img):
    # 均值滤波
    return cv2.blur(img, (5, 5))


def get_medianBlur(img):
    # 中值滤波
    return cv2.medianBlur(img, 5)


def get_GaussianBlur(img):
    # 高斯滤波
    return cv2.GaussianBlur(img, (5, 5), 0)


def get_Canny(img):
    # 高斯边缘检测
    return cv2.Canny(img, 0, 50)


def get_erode(img):
    # 腐蚀
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion


def get_dilate(img):
    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation


def get_open_morph(img):
    # 开运算
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def opencv_channel_split_merge(img):
    r, g, b = cv2.split(img)  # 图像的拆分，将彩色图像划分为三种颜色
    img23 = cv2.merge([r, g, b])  # 将三种颜色通道的图片融合

def get_close_morph(img):
    # 闭运算 先膨胀后腐蚀
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def get_gradient(img):
    # 形态学梯度 图像的膨胀和腐蚀之间的差异，结果看起来像目标的轮廓
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient


def get_tophat(img):
    # 顶帽 原图像与开运算图的区别，突出原图像中比周围亮的区域
    kernel = np.ones((5, 5), np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    return tophat


def get_blackhat(img):
    # 黑帽 闭运算图 - 原图像,突出原图像中比周围暗的区域
    kernel = np.ones((5, 5), np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return blackhat


"""
# Rectangular Kernel
cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

Out[4]:
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)

# Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
Out[5]:
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)

# Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
Out[6]:
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
"""


def print_all_feature(save_dir, image):
    """打印所有的opencv特征图片
        Args:image是灰度图
    """
    save_dir = os.path.join(save_dir, "log_feature")

    resetDir(save_dir)
    cv2.imwrite(os.path.join(save_dir, "origin.jpg"), image)

    # 超特征搜索
    hy_params = [
        {"name": ["scharrx64", "scharry64", "scharrxy64"], "func": "get_scharr"},
        {"name": ["adaptive_threshold"], "func": "get_adaptive_threshold"},
        {"name": ["threshold"], "func": "get_threshold"},
        {"name": ["sobelx", "sobely", "sobelxy"], "func": "get_sobel"},
        {"name": ["laplace"], "func": "get_laplace"},
        {"name": ["blur"], "func": "get_blur"},
        {"name": ["medianBlur"], "func": "get_medianBlur"},
        {"name": ["GaussianBlur"], "func": "get_GaussianBlur"},
        {"name": ["Canny"], "func": "get_Canny"},
        {"name": ["profile"], "func": "graytoprofile"},
        {"name": ["erode"], "func": "get_erode"},
        {"name": ["dilate"], "func": "get_dilate"},
        {"name": ["open_morph"], "func": "get_open_morph"},
        {"name": ["close_morph"], "func": "get_close_morph"},
        {"name": ["gradient"], "func": "get_gradient"},
        {"name": ["tophat"], "func": "get_tophat"},
        {"name": ["blackhat"], "func": "get_blackhat"},
    ]

    def hy_func(func=None, name_list=[], result=None, now_layers_num=1, layers_num=2):
        """

        Args:
            func:
            name_list:
            result:
            now_layers_num: 当前的层数
            layers_num: 总层数

        Returns:

        """
        assert layers_num < 3 # 最多找3层，因为计算量特大
        for hy2 in hy_params:  # 第一层
            names2 = hy2["name"]
            func2 = hy2["func"]
            if (func is None) or (func is not None and func != func2):

                results2 = globals()[func2](result)
                if len(names2) == 1:
                    results2 = [results2]
                for name2, result2 in zip(names2, results2):
                    if len(name_list) == 0:
                        filename = "{}.jpg".format(name2)
                    else:
                        filename = "{}_{}.jpg".format("_".join(name_list), name2)

                    cv2.imwrite(os.path.join(save_dir, filename), result2)

                    if layers_num > now_layers_num:
                        name_list2 = copy.deepcopy(name_list)
                        name_list2.append(name2)
                        hy_func(func2, name_list2, result2, now_layers_num + 1, layers_num)  # 下一层

    hy_func(result=image)


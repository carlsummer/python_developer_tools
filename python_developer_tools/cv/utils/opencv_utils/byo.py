# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/2/2021 12:56 PM
# @File:byo
import os

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

def get_scharr(o,ddepth=cv2.CV_64F):
    """获取scharr的特征"""
    scharrx = cv2.Scharr(o, ddepth, 1, 0)
    scharry = cv2.Scharr(o, ddepth, 0, 1)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.convertScaleAbs(scharry)

    scharrxy = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    return scharrx, scharry, scharrxy


def get_sobel(img,ddepth=cv2.CV_64F):
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

    adaptive_threshold = get_adaptive_threshold(image)
    cv2.imwrite(os.path.join(save_dir, "adaptive_threshold.jpg"), adaptive_threshold)

    scharrx64, scharry64, scharrxy64 = get_scharr(image)
    cv2.imwrite(os.path.join(save_dir, "scharrx64.jpg"), scharrx64)
    cv2.imwrite(os.path.join(save_dir, "scharrx64_threshold.jpg"), get_threshold(scharrx64))
    cv2.imwrite(os.path.join(save_dir, "scharry64.jpg"), scharry64)
    cv2.imwrite(os.path.join(save_dir, "scharry64_threshold.jpg"), get_threshold(scharry64))
    cv2.imwrite(os.path.join(save_dir, "scharrxy64.jpg"), scharrxy64)
    cv2.imwrite(os.path.join(save_dir, "scharrxy64_threshold.jpg"), get_threshold(scharrxy64))

    scharrx16, scharry16, scharrxy16 = get_scharr(image,cv2.CV_16S)
    cv2.imwrite(os.path.join(save_dir, "scharrx16.jpg"), scharrx16)
    cv2.imwrite(os.path.join(save_dir, "scharrx16_threshold.jpg"), get_threshold(scharrx16))
    cv2.imwrite(os.path.join(save_dir, "scharry16.jpg"), scharry16)
    cv2.imwrite(os.path.join(save_dir, "scharry16_threshold.jpg"), get_threshold(scharry16))
    cv2.imwrite(os.path.join(save_dir, "scharrxy16.jpg"), scharrxy16)
    cv2.imwrite(os.path.join(save_dir, "scharrxy16_threshold.jpg"), get_threshold(scharrxy16))

    sobelx, sobely, sobelxy = get_sobel(image)
    cv2.imwrite(os.path.join(save_dir, "sobelx.jpg"), sobelx)
    cv2.imwrite(os.path.join(save_dir, "sobely.jpg"), sobely)
    cv2.imwrite(os.path.join(save_dir, "sobelxy.jpg"), sobelxy)

    laplace = get_laplace(image)
    cv2.imwrite(os.path.join(save_dir, "laplace.jpg"), laplace)

    blur = get_blur(image)
    cv2.imwrite(os.path.join(save_dir, "blur.jpg"), blur)

    medianBlur = get_medianBlur(image)
    cv2.imwrite(os.path.join(save_dir, "medianBlur.jpg"), medianBlur)

    GaussianBlur = get_GaussianBlur(image)
    cv2.imwrite(os.path.join(save_dir, "GaussianBlur.jpg"), GaussianBlur)

    Canny = get_Canny(GaussianBlur)
    cv2.imwrite(os.path.join(save_dir, "Canny.jpg"), Canny)

    profile = graytoprofile(image)
    cv2.imwrite(os.path.join(save_dir, "profile.jpg"), profile)

    erode = get_erode(image)
    cv2.imwrite(os.path.join(save_dir, "erode.jpg"), erode)

    dilate = get_dilate(image)
    cv2.imwrite(os.path.join(save_dir, "dilate.jpg"), dilate)

    open_morph = get_open_morph(image)
    cv2.imwrite(os.path.join(save_dir, "open_morph.jpg"), open_morph)

    close_morph = get_close_morph(image)
    cv2.imwrite(os.path.join(save_dir, "close_morph.jpg"), close_morph)

    gradient = get_gradient(image)
    cv2.imwrite(os.path.join(save_dir, "gradient.jpg"), gradient)

    # 超特征搜索
    # hy = []
    # hy_num =

    tophat = get_tophat(image)
    cv2.imwrite(os.path.join(save_dir, "tophat.jpg"), tophat)

    blackhat = get_blackhat(image)
    cv2.imwrite(os.path.join(save_dir, "blackhat.jpg"), blackhat)

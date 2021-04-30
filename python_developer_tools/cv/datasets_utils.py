# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/30/2021 9:28 AM
# @File:datasets_utils
import random
import math
import cv2
import numpy as np
from imgaug import augmenters as iaa  # 引入数据增强的包


def resize_image(img, resize_size=[640, 640]):
    """对图片进行resize"""
    h0, w0 = img.shape[:2]  # origin hw
    # rh0 = self.opt.img_size[0] / h0  # resize image to img_size
    # rw0 = self.opt.img_size[1] / w0  # resize image to img_size
    r = max(resize_size[0], resize_size[1]) / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        # INTER_NEAREST	最近邻插值
        # cv2.INTER_LINEAR  双线性插值（默认设置）
        # cv2.INTER_AREA 使用像素区域关系进行重采样。 它可能是图像抽取的首选方法，因为它会产生无云纹理的结果。 但是当图像缩放时，它类似于INTER_NEAREST方法。
        # img = cv2.resize(img, (int(w0 * rw0), int(h0 * rh0)), interpolation=cv2.INTER_LINEAR)
        # interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
    return img


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    """直方图增强"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def imgaug_augmenters(img, hyp):
    """imgaug库增强"""
    seq = iaa.Sequential([
        # Sometimes是指指针对50%的图片做处理
        iaa.Sometimes(
            hyp["iaa_Gaussian_r"],
            # 高斯模糊
            iaa.GaussianBlur(sigma=(hyp["iaa_Gaussian_sigma_1"], hyp["iaa_Gaussian_sigma_2"]))
        ),
        # 使用随机组合上面的数据增强来处理图片
    ], random_order=True)
    images_aug = seq.augment_image(img)  # 是处理多张图片augment_images
    return images_aug


def cutout(img, num_holes=1):
    """
    其思想也很简单，就是对训练图像进行随机遮挡，该方法激励神经网络在决策时能够更多考虑次要特征，而不是主要依赖于很少的主要特征，如下图所示：
    Randomly mask out one or more patches from an image.
    """
    h, w, _ = img.shape

    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16
    s = random.choice(scales)
    for _ in range(num_holes):
        y = np.random.randint(h)
        x = np.random.randint(w)

        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        y1 = np.clip(max(0, y - mask_h // 2), 0, h)
        y2 = np.clip(max(0, y + mask_h // 2), 0, h)
        x1 = np.clip(max(0, x - mask_w // 2), 0, w)
        x2 = np.clip(max(0, x + mask_w // 2), 0, w)

        # apply random color mask
        img[y1: y2, x1: x2, :] = [random.randint(64, 191) for _ in range(3)]
    return img


def random_perspective(img, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective 透视图
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale 旋转和缩放
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear 剪切
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    return img

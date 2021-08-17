# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/16/2021 4:19 PM
# @File:line_imgaug
import copy
import random

import cv2
import math
import numpy as np

def cutout(image, labels):
    """
    其思想也很简单，就是对训练图像进行随机遮挡，该方法激励神经网络在决策时能够更多考虑次要特征，而不是主要依赖于很少的主要特征，如下图所示：
    Randomly mask out one or more patches from an image.
    """
    gap = 10

    image_copy = copy.deepcopy(image)

    # for (a, b) in labels:
    #     _ = cv2.line(image_copy, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), thickness=1)

    h, w = image_copy.shape[:2]

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        # s = random.choice(scales)
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        if not ( (xmin-gap <= labels[0,:,0][0] <=xmax+gap) or ((xmin-gap<= labels[0,:,0][1] <=xmax+gap)) or
                 (xmin-gap<= labels[1,:,0][0] <=xmax+gap) or ((xmin-gap<= labels[1,:,0][1] <=xmax+gap)) ):
            image_copy[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

    return image_copy

def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    img_copy = copy.deepcopy(img)

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
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

    # Transform label coordinates
    #targets = targets[0]

    n = len(targets)
    first_targets = []
    first_targets.append(targets[0])
    targets = np.array(targets)
    xy = np.ones((n * 4, 3))
    #xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(1 * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = xy @ M.T  # transform
    xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    new_copy = copy.deepcopy(new)

    # clip
    new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
    new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

    is_reasonable = False
    if ((new == new_copy).all()):
        is_reasonable = True

    # new_flatten = new.flatten()
    # for i in range(len(new)):
    #     if i % 2 == 0 and (new_flatten[i] < 0 or new_flatten[i] > width - 1):
    #         is_reasonable = False
    #     elif i % 2 != 0 and (new_flatten[i] < 0 or new_flatten[i] > height - 1):
    #         is_reasonable = False

    if is_reasonable:
        return img, new, is_reasonable
    else:
        return img_copy, targets, is_reasonable

    # xy = np.ones((4, 3))
    # targets_2 = [targets[0] * width, 0, targets[1] * width, height, targets[0] * width, height, targets[1] * width,
    #              0]  # x1y1, x2y2, x1y2, x2y1
    # targets_3 = np.array(targets_2)
    # xy[:, :2] = targets_3.reshape(4, 2)
    # xy = xy @ M.T  # transform
    # xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(1, 8)  # perspective rescale or affine
    # x = xy[:, [0, 2, 4, 6]]
    # y = xy[:, [1, 3, 5, 7]]

    # new_labels = [x[0][2],x[0][1]]
    # if 10<new_labels[0]<width-10 and 10 < new_labels[1] < width-10:
    #     return img,[min(new_labels).clip(0, width) / width,max(new_labels).clip(0, width) / width]
    # else:
    #     return img, targets
# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/30/2021 9:26 AM
# @File:matplotlib_utils
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import random
import cv2


def plot_labels(labels, save_dir='', imgname="labels"):
    # plot dataset labels
    c = labels  # classes, boxes
    nc = max(c) + 1  # number of classes
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    ax.hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax.set_xlabel('classes')
    ax.set_ylabel("number")
    plt.savefig(Path(save_dir) / "{}_{}".format(imgname, 'labels.png'), dpi=200)
    plt.close()


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, [0, 255, 255], thickness=tl, lineType=cv2.LINE_AA)
    if label:
        textFontsize = 10
        tf = max(textFontsize - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=textFontsize / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        # cv2.rectangle(img, c1, c2, [0, 0, 255], -1, cv2.LINE_AA)  # filled
        textStartx = int((c1[0] + (x[2] - x[0]) / 2) - len(label) * (textFontsize / 3) * tf)
        cv2.putText(img, label, (textStartx, c1[1] - 10), 0, textFontsize / 3, [0, 0, 225], thickness=tf,
                    lineType=cv2.LINE_AA)

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
from scipy import signal

def plt_show_histogram(histogram):
    """将list或者nparray画成图 画直方图"""
    hist_x=get_histogram_hist_x(histogram)
    plt.plot(hist_x, histogram)
    plt.show()
    plt.close()

def plt_save_histogram(histogram,save_dir='', imgname="labels"):
    hist_x=get_histogram_hist_x(histogram)
    plt.plot(hist_x, histogram)
    plt.savefig(Path(save_dir) / "{}_{}".format(imgname, 'labels.png'), dpi=200)
    plt.close()

def _listornparray_2_plt(row_histogram):
    """将list或者nparray画成图 画直方图"""
    lenth = len(row_histogram)
    plt.figure(1)
    hist_x = np.linspace(0, lenth - 1, lenth)
    plt.title("row_histogram")
    plt.rcParams['figure.figsize'] = (lenth, 8)  # 单位是inches

    x_major_locator = plt.MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)  # x轴按1刻度显示

    plt.plot(hist_x, row_histogram)
    higher_q = np.max(row_histogram) * (5 / 8)
    plt.plot([0, lenth - 1], [higher_q, higher_q])
    num_peak_3 = signal.find_peaks(row_histogram, distance=1)  # distance表极大值点的距离至少大于等于10个水平单位
    for ii in range(len(num_peak_3[0])):
        if (row_histogram[num_peak_3[0][ii]] > np.mean(row_histogram)) and (
                row_histogram[num_peak_3[0][ii]] != np.max(row_histogram)
        ) and (
                num_peak_3[0][ii] > 10
        ):
            plt.plot(num_peak_3[0][ii], row_histogram[num_peak_3[0][ii]], '*', markersize=10)
            plt.axvline(num_peak_3[0][ii])
            print(num_peak_3[0][ii])

    plt.savefig("row_histogram_peak.jpg")
    plt.close()

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

def get_histogram_hist_x(histogram):
    lenth = len(histogram)
    hist_x = np.linspace(0, lenth - 1, lenth)
    return hist_x

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

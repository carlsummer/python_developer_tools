# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/16/2021 1:34 PM
# @File:metrology-demo.py
# !/usr/bin/env python3

import sys
import argparse
import numpy as np
import cv2 as cv
from python_developer_tools.cv.utils.matplotlib_utils import plt_show_histogram
import scipy.ndimage


if __name__ == '__main__':
    # command line argument parsing
    # change defaults here

    parser = argparse.ArgumentParser()
    parser.add_argument("--picture", dest="fname", metavar="PATH", type=str, default="sdf11.jpg",
                        help="path to picture file")
    parser.add_argument("--sigma", type=float, default=2.0, metavar="PX",
                        help="sigma for gaussian lowpass on sampled signal, before gradient is calculated")
    args = parser.parse_args()

    ########## here be dragons ##########
    args.stride = 1

    im = cv.imread(args.fname, cv.IMREAD_GRAYSCALE)
    imh, imw = im.shape[:2]
    im = 255 - im
    im = im.astype(np.float32)  # * np.float32(1/255)
    # pick one
    samples = np.sum(im, axis=1)
    plt_show_histogram(samples)
    # smoothing to remove noise
    samples = scipy.ndimage.gaussian_filter1d(samples, sigma=args.sigma / args.stride)

    # off-by-half in position because for values [0,1,1,0] this returns [+1,0,-1]
    gradient = np.diff(samples) / args.stride

    i_falling = np.argmin(gradient)  # in samples
    i_rising = np.argmax(gradient)  # in samples

    distance = np.abs(i_rising - i_falling) * args.stride  # in pixels

    print(f"distance: ",distance)


    plt_show_histogram(gradient)


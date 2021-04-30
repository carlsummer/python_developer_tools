# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/30/2021 9:26 AM
# @File:matplotlib_utils
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def plot_labels(labels, save_dir='',imgname="labels"):
    # plot dataset labels
    c = labels  # classes, boxes
    nc = max(c) + 1  # number of classes
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), tight_layout=True)
    ax.hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax.set_xlabel('classes')
    ax.set_ylabel("number")
    plt.savefig(Path(save_dir) / "{}_{}".format(imgname,'labels.png'), dpi=200)
    plt.close()
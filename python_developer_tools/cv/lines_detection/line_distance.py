# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/24/2021 8:56 AM
# @File:line_distance
# 求两条直线的距离
import matplotlib.pylab as plt
import numpy as np
from numpy import array, int32
from typing import List
from itertools import combinations


def show_lines_data(lines: List, label: str):
    # plt.xlim(200, 650)
    # plt.ylim(160, 230)
    if len(lines) == 1:
        for x1, y1, x2, y2 in lines:
            plt.suptitle(label)
            plt.plot([x1, x2], [y1, y2])
    else:
        for line in lines:
            # if isinstance()
            for x1, y1, x2, y2 in line:
                plt.suptitle(label)
                plt.plot([x1, x2], [y1, y2])
    plt.show()


def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return int(k/ncol), int(k%ncol)

def get_two_line_short_length(point1, point2):
    x1, y1, n = get_x_y(point1)
    x2, y2, m = get_x_y(point2)
    ar = np.zeros((n, m))  # 11*11矩阵
    for i in range(n):  # 欧氏距离
        ar[i, :] = np.sqrt((x2 - x1[i]) ** 2 + (y2 - y1[i]) ** 2)
    x2_index,x1_index = find_min_idx(ar)
    return x1[x1_index],y1[x1_index],x2[x2_index],y2[x2_index],ar.min() # 取最小的


def get_x_y(point):
    g = point[1] - point[3]  # y1 - y2
    h = point[0] - point[2]  # x1 - x2 #  这个始终为负(point始终在左边)
    if g < 0 or h < 0:
        t = array([-i / 10 for i in range(0, 100)])  # 斜率为负数, 这里分的越长，精度越高
    else:
        t = array([i / 10 for i in range(0, 100)])  # 斜率不存在
    x = array(point[0]) + h * t  # 线段横坐标平均取11个点
    y = array(point[1]) + g * t  # 线段纵坐标平均取11个点
    return x, y, len(t)

#  线段分簇算法
def count_line_cluster(lines, r: int) -> int:
    copy_line = sorted(lines, key=lambda x: x[0][0])  # 按照横坐标从左排到右
    count_list = [copy_line[0]] # 第一个肯定是一簇中的某一条
    for j in range(len(copy_line) - 1): # 遍历主列表
        for i in range(len(count_list)): # 遍历记数列表
            if get_two_line_short_length(copy_line[j + 1][0], count_list[i][0]) > r:  # 如果主列表中线段与记数列表之间的距离大于r
                if len(count_list) == i + 1:  # 判断该记数数组中是否已经含有该簇的线段，等于说明没有该簇的线段
                    count_list.append(copy_line[j + 1]) # 加入该簇的线段
                    break # 跳出 记数列表循环， 进入一个主循环
            else:
                if copy_line[j + 1][0][2] > count_list[i][0][2]: # 进入里面说明记数列表已经含有该簇的线段，但是还要更新一下该簇的线段，为了取得才簇在中最长的线段，这样才能使该簇其他较短的线段与该簇最长线段在r内
                    count_list[i] = copy_line[j + 1] # 更新记数列表中该簇的线段
                    break # 跳出记数列表循， 进入一个主循环
                else:
                    break# 跳出记数列表循， 进入一个主循环

    return len(count_list)


if __name__ == '__main__':
    short_length = get_two_line_short_length(point1=np.array([230, 114,3565,7]), point2=np.array([227,200,3726,162])) # x1,y1,x2,y2
    print(short_length)
    # lines = [array([[ 21, 225, 105, 225]], dtype=int32), array([[416, 166, 458, 166]], dtype=int32), array([[484, 227, 600, 227]], dtype=int32), array([[ 95, 174, 232, 169]], dtype=int32), array([[ 98, 172, 201, 169]], dtype=int32), array([[ 38, 224, 187, 222]], dtype=int32), array([[484, 226, 572, 225]], dtype=int32), array([[104, 222, 187, 221]], dtype=int32)]
    # show_lines_data(lines, label='lines')
    # print(count_line_cluster(lines, r=30))



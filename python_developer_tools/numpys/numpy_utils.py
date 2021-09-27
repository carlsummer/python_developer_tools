# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/1/2021 3:08 PM
# @File:numpy_utils
import numpy as np

# numpy 按维度合并
# lsd_mask = np.zeros((512, 64), np.uint8)
# img = np.concatenate((lsd_mask[:,:,None],lsd_mask[:,:,None],lsd_mask[:,:,None]),axis=2)

# numpy 过滤
# distanceIds = np.where((dataset > self.distance_trod_top) | (dataset < self.distance_trod_bottom))
# return np.array(self.pian_problems)[distanceIds].tolist()

def argsort2d(arr):
    # 对numpy 数组按大小排序，返回数字大的位置，数字越大越后，
    # eg: arr=[[2,2,2]
        # ,[2,3,2]
        # ,[2,5,2]
        # ,[2,2,2]]
    # 返回 [[4,2],.....[2,2],[3,2]]
    return np.dstack(np.unravel_index(np.argsort(arr.ravel()), arr.shape))[0]

def np_softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def get_mode(distancelist=[6, 7, 7, 6, 7, 8, 8, 4, 6, 5, 5,
                           5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4,
                           5, 5, 5, 5, 5, 7, 7, 8, 8, 6, 6, 5, 5, 5,
                           4, 6, 6, 7, 7, 7, 4, 5, 5, 5, 5, 5]):
    """求list的众数"""
    counts = np.bincount(np.array(distancelist))
    label_classnum = np.argmax(counts)  # 众数为classnum对应label中的分类
    return label_classnum

def remove_discrete_values_quantile_ms(distancelist=[6, 7, 7, 6, 7, 8, 8, 4, 6, 5, 5,
                           5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4,
                           5, 5, 5, 5, 5, 7, 7, 8, 8, 6, 6, 5, 5, 5,
                           4, 6, 6, 7, 7, 7, 4, 5, 5, 5, 5, 5],iqr_throld=1.5):
    """采用标准差法去除离散值"""
    dnp = np.array(distancelist)
    xbar = np.mean(dnp)
    xstd = np.std(dnp)
    dnpnew = dnp[np.where((dnp  < xbar + iqr_throld * xstd ) & (dnp > xbar - iqr_throld * xstd))]
    return dnpnew

def remove_discrete_values_quantile(distancelist=[6, 7, 7, 6, 7, 8, 8, 4, 6, 5, 5,
                           5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 4,
                           5, 5, 5, 5, 5, 7, 7, 8, 8, 6, 6, 5, 5, 5,
                           4, 6, 6, 7, 7, 7, 4, 5, 5, 5, 5, 5],iqr_throld=1.5):
    """采用箱线图法去除离散值"""
    dnp = np.array(distancelist)
    Q1 = np.quantile(dnp, q=0.25)
    Q3 = np.quantile(dnp, q=0.75)
    IQR = Q3 - Q1
    dnpnew = dnp[np.where((dnp < Q3 + iqr_throld * IQR) & (dnp > Q1 - iqr_throld * IQR))]
    return dnpnew
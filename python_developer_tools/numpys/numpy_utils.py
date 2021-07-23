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


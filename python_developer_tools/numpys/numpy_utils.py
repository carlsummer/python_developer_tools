# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/1/2021 3:08 PM
# @File:numpy_utils
import numpy as np

# numpy 按维度合并
lsd_mask = np.zeros((512, 64), np.uint8)
img = np.concatenate((lsd_mask[:,:,None],lsd_mask[:,:,None],lsd_mask[:,:,None]),axis=2)

# numpy 过滤
distanceIds = np.where((dataset > self.distance_trod_top) | (dataset < self.distance_trod_bottom))
# return np.array(self.pian_problems)[distanceIds].tolist()
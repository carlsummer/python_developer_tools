# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/11/2021 9:37 AM
# @File:list_utils
def find_all_index(arr, item):
    # 查询是否存在并且并返回下标
    # print(find_all_index([1,2,3,4,4,3,89],4))
    return [i for i, a in enumerate(arr) if a == item]
# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/4/2021 8:41 AM
# @File:metrics

def get_distance_acc(l1loss,px=64):
    """获取距离准确率"""
    return 1-(1/px)*l1loss if l1loss<px else 0
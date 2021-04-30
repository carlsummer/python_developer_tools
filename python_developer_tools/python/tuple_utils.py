# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 8:35 PM
# @File:data_utils
def getkeys_by_tuple(tuplemm):
    """返回元组中所有的key"""
    return [tuple_tmp[0] for tuple_tmp in tuplemm]


def tuple_value_by_key(tuplemm, key):
    """
    根据key获取元组中value的值
    """
    for tuple_tmp in tuplemm:
        if tuple_tmp[0] == key:
            return tuple_tmp[1]


def tuple_key_by_value(tuplemm, value):
    """
    根据value获取元组中key的值
    """
    for tuple_tmp in tuplemm:
        if tuple_tmp[1] == value:
            return tuple_tmp[0]

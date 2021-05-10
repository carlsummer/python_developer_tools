# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 8:38 PM
# @File:obj_utils
from python_developer_tools.python.string_utils import str_is_null


def obj_is_null(obj):
    """判断对象是否为空"""
    if obj is None:
        return True
    if isinstance(obj, list) and len(obj) == 0:
        return True
    if isinstance(obj, str):
        return str_is_null(obj)

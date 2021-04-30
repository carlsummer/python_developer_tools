# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 8:36 PM
# @File:string_utils
def str_is_null(strings):
    """判断字符串是否为空"""
    if strings is None:
        return True
    if strings.strip() == '':
        return True  # 是空
    return False

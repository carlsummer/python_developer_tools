# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/13/2021 1:54 PM
# @File:testser
import ctypes


def get_id(value):
    return id(value)


def get_val_by_id(address):
    return ctypes.cast(address, ctypes.py_object).value  # 读取地址中的变量


def read_id_val(value):
    # value = 'hello world'  定义一个字符串变量
    address = get_id(value)  # 获取value的地址，赋给address
    get_value = get_val_by_id(address)
    print(address, get_value)


if __name__ == '__main__':
    read_id_val("hello world")

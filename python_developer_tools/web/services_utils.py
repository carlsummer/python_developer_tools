# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/10/2021 2:50 PM
# @File:services_utils
import time


def get_items_serial_number(prefix, obj):
    """获取物品唯一货号"""
    date_str = time.strftime("%Y%m%d", time.localtime(time.time()))
    # 生成一个固定6位数的流水号
    objmodel = obj.objects.last()
    serial_number = objmodel.id if objmodel else 0
    serial_number = "{0:06d}".format(serial_number + 1)
    return "{}{}{}".format(prefix, date_str, serial_number)


if __name__ == '__main__':
    get_items_serial_number("items", 32)

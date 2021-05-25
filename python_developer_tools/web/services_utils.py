# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/10/2021 2:50 PM
# @File:services_utils
import random
import time
from random import choice

def get_items_serial_number(prefix, obj):
    """获取物品唯一货号"""
    date_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
    # 生成一个固定6位数的流水号
    objmodel = obj.objects.last()
    serial_number = objmodel.id if objmodel else 0
    serial_number = "{0:06d}".format(serial_number + 1)
    return "{}{}{}".format(prefix, date_str, serial_number)

def generate_code():
    """
    生成四位数字的验证码
    :return:
    """
    seeds = "1234567890"
    random_str = []
    for i in range(4):
        random_str.append(choice(seeds))

    return "".join(random_str)

def make_code():
    """
    :return: code 6位随机数
    """
    code = ''
    for item in range(6):
        code += str(random.randint(0, 9))
    return code

if __name__ == '__main__':
    get_items_serial_number("items", 32)

# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/24/2021 12:40 PM
# @File:os调用shell命令并且获取返回值
import os

val = os.popen('find /home/admin/mediaimgs-haining/ -name "234235351*"|grep origin')
vallist = [i for i in val.readlines()]

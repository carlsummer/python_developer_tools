# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 8:41 PM
# @File:time_utils
import datetime
import time


def get_time_stamp():
    """获取毫秒级的时间"""
    ct = time.time()
    local_time = time.localtime(ct)
    data_head = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
    data_secs = (ct - int(ct)) * 1000
    time_stamp = "%s.%03d" % (data_head, data_secs)
    return time_stamp

def str2datetime(dd):
    """
    # str转时间格式：
    dd = '2019-03-17 11:00:00'
    dd = datetime.datetime.strptime(dd, "%Y-%m-%d %H:%M:%S")
    print(dd,type(dd))
    """
    return datetime.datetime.strptime(dd, "%Y-%m-%d %H:%M:%S")

def datetime2str(mtime):
    """datetime格式转str"""
    return mtime.strftime("%Y-%m-%d %H:%M:%S")
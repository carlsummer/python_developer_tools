# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/13/2021 11:23 AM
# @File:response_utils

# 编写视图函数，进行设置
from datetime import datetime,timedelta

def set_cookie(response,key,value):
    """设置cookie"""
    ''' max_age 设置过期时间，单位是秒 '''
    # response.set_cookie('name', 'tong', max_age=14 * 24 * 3600)
    ''' expires 设置过期时间，是从现在的时间开始到那个时间结束 '''
    response.set_cookie(key, value, expires=datetime.now()+timedelta(days=14))
    return response
# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/13/2021 4:49 PM
# @File:json_utils
import json

def str_2_json(str):
    # str = '{"key": "wwww", "word": "qqqq"}'
    j = json.loads(str)
    # print(j)
    # print(type(j))
    return j
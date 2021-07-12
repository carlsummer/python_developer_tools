# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/3/2021 4:23 PM
# @File:json_utils
import json


def str_2_json(str):
    # str = '{"key": "wwww", "word": "qqqq"}'
    j = json.loads(str)
    # print(j)
    # print(type(j))
    return j


def read_json_file(filepath):
    with open(filepath, 'r', encoding="utf-8") as load_f:
        load_dict = json.load(load_f)
    return load_dict

def save_json_file(filepath,datas):
    # datas = ['joker', 'joe', 'nacy', 'timi']
    with open(filepath,'w') as file_obj:
        json.dump(datas,file_obj)
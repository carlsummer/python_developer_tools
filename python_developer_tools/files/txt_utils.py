# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 4:18 PM
# @File:txt_utils
import ast

def save_str_txt(strings,save_path):
    with open(save_path, 'a', encoding='utf-8') as f:
        f.write(strings)  # label format

def save_jsondata_txt(json, save_path):
    # 将内容保存为txt
    if json is not None:
        with open(save_path, 'a+', encoding='utf-8') as f:
            f.write(str(json))


def read_predict_txt(txt_path):
    # 读取txt中的内容
    with open(txt_path, 'r') as f:
        file_context = f.read()
    predict_dict = ast.literal_eval(file_context)
    return predict_dict

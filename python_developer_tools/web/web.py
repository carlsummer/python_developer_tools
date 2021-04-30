# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:4/29/2021 8:39 PM
# @File:web.py
import os

import requests
import socket
from xlrd import open_workbook
from xlutils.copy import copy


def getiphostname():
    # 获取计算机名称
    hostname = socket.gethostname()
    # 获取本机IP
    ip = socket.gethostbyname(hostname)
    return ip, hostname


def post_get_data(url='http://127.0.0.1:8000/pv/api/', data={"imageName": "sdfsdfasd"}):
    """发送post请求通过表单列的方式并获取数据"""
    r = requests.post(url, data=data)
    return r.text


def post_get_data_by_json(url='http://127.0.0.1:8000/pv/api/', json={"imageName": "sdfsdfasd"}):
    """发送post请求通过json并获取数据"""
    r = requests.post(url, json=json)
    return r.text


def get_request_data(url):
    """get请求获取数据"""
    r = requests.get(url)
    return r.text


def export_excel(values=["1", "2", "3"]):
    """读取xls模板并写入数据"""
    export_excel_dir = os.path.join("", "")
    if not os.path.exists(export_excel_dir):
        os.makedirs(export_excel_dir)
    filename = "统计报表.xls"
    xlstemplates = os.path.join(export_excel_dir, 'tongji.xls')
    reportSavePath = os.path.join(export_excel_dir, filename)

    rexcel = open_workbook(xlstemplates, formatting_info=True)  # 用wlrd提供的方法读取一个excel文件 打开excel，保留文件格式
    rows = rexcel.sheets()[0].nrows  # 用wlrd提供的方法获得现在已有的行数
    excel = copy(rexcel)  # 用xlutils提供的copy方法将xlrd的对象转化为xlwt的对象
    table = excel.get_sheet(0)  # 用xlwt对象的方法获得要操作的sheet

    row = rows
    for value in values:
        table.write(row, 0, value)  # xlwt对象的写方法，参数分别是行、列、值
        table.write(row, 1, "haha")
        table.write(row, 2, "lala")
        row += 1
    excel.save(reportSavePath)  # xlwt对象的保存方法，这时便覆盖掉了原来的excel
    return reportSavePath, filename

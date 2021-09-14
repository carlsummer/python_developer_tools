# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/10/2021 2:10 PM
# @File:test-python
import time
t = time.time()
for i in range(10**8):
    continue
print(time.time() - t)
# python test-python.py 3.5080716609954834s


# ./pypy ~/workspace/python_developer_tools/python_developer_tools/python/加速/test-python.py 0.1419203281402588
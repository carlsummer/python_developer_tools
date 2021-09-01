# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/1/2021 10:28 AM
# @File:pymem_utils
from pymem import *

Pymem = Pymem("PlantsVsZombies.exe")

addr = Pymem.read_int(0x6A9EC0)

addr = Pymem.read_int(addr + 0x768)

v1 = Pymem.read_int(addr + 0x5560)

print(f"现有阳光数值:{v1}")

v2 = int(input("输入要改的数值:"))

Pymem.write_int(addr + 0x5560, v2)

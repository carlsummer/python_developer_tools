# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:10/20/2021 10:15 AM
# @File:正则re
import re
# cjj_0_1_1_6509539287014846_1#22.jpg  -> 提取 1和22
print(re.findall(r'.*_(\d+)#(\d+)', 'cjj_0_1_1_6509539287014846_1#22.jpg'))
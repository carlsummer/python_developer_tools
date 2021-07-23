# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/22/2021 12:53 PM
# @File:sdf

"""
python atexit 模块定义了一个 register 函数，用于在 python 解释器中注册一个退出函数，
这个函数在解释器正常终止时自动执行,一般用来做一些资源清理的操作。
atexit 按注册的相反顺序执行这些函数; 例如注册A、B、C，在解释器终止时按顺序C，B，A运行。
Note：如果程序是非正常crash，或者通过os._exit()退出，注册的退出函数将不会被调用。
"""
import os
from atexit import register


def main():
    print('Do something.')


@register
def _atexit():
    print('Done.')


def goodbye(name, adjective):
    print('Goodbye, %s, it was %s to meet you.' % (name, adjective))


if __name__ == '__main__':
    register(goodbye, adjective='nice', name='Donny')
    main()
    # exit(0)  # 程序退出了

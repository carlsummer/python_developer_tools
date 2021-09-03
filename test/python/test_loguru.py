# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/3/2021 8:48 AM
# @File:loguru
from loguru import logger
# /home/zengxh/anaconda3/envs/CreepageDistance/bin/pip install loguru

@logger.catch
# Traceback 记录
def main():
    print("start")
    h = 2/0
    print("end")

if __name__ == '__main__':
    #https://cuiqingcai.com/7776.html

    # main()

    # logger.add('runtime.log') # 将所有的日志都输出到log文件
    # logger.debug('this is a debug message')

    # 我们想一天输出一个日志文件，或者文件太大了自动分隔日志文件
    # logger.add('runtime_{time}.log', rotation="500 MB") #实现每 500MB 存储一个文件
    # logger.add('runtime_{time}.log', rotation='00:00') #实现每天 0 点新创建一个 log 文件输出
    # logger.add('runtime_{time}.log', rotation='1 week') #一周创建一个 log 文件
    # logger.add('runtime.log', retention='10 days') # 保留最新 10 天的 log

    #loguru 还可以配置文件的压缩格式，比如使用 zip 文件格式保存
    # logger.add('runtime.log', compression='zip')
    pass




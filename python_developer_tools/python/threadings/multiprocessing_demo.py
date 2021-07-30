# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/30/2021 10:40 AM
# @File:multiprocessing_demo
import multiprocessing
import time

from multiprocessing.managers import BaseManager

class CA(object):
    def __init__(self):
        self.name ="A"
        self.valuey = "a_value"
    def run(self):
        self.valuey = "2_value"
    def prints(self):
        return self.valuey

def func1(sleep,sharearg,shareca,q):
    time.sleep(sleep)
    ct = time.time()
    print("func1结束{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))
    # sharearg = multiprocessing.Manager().Value("s","sharearg")
    # sharearg.value
    sharearg.value = sharearg.value + "func1"
    shareca.value.run()
    q.put([{"name": "func1","sharearg":sharearg.value,"shareca":shareca.value.prints()}])

def func2(sleep,sharearg,shareca,q):
    time.sleep(sleep)
    ct = time.time()
    print("func2结束{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))
    sharearg.value = sharearg.value + "func2"
    q.put([{"name": "func2","sharearg":sharearg.value}])

def func3(sleep,sharearg,shareca):
    time.sleep(sleep)
    ct = time.time()
    print("func3结束{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))
    sharearg.value = sharearg.value + "func3"
    return [{"name": "func3","sharearg":sharearg.value}]

if __name__ == '__main__':
    start = time.time()
    results = []

    q = multiprocessing.Queue() # 获取返回结果的队列
    manager = multiprocessing.Manager()
    sharearg = manager.Value("s","sharearg")
    ca = CA()
    shareca = manager.Value(CA, ca) # 共享了类但是不会改变值
    shareca.value.run()
    print(shareca.value.valuey)

    ct = time.time()
    print("开始{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))

    thread_func1 = multiprocessing.Process(target=func1, args=(0.4,sharearg,shareca, q))
    thread_func2 = multiprocessing.Process(target=func2, args=(0.2,sharearg,shareca, q))

    # 启动线程
    thread_func1.start()
    thread_func2.start()

    results.extend(func3(0.9,sharearg,shareca))

    thread_func1.join()
    thread_func2.join()

    results.extend(q.get())
    results.extend(q.get())


    print('Main thread has ended!',time.time()-start,results)


# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/30/2021 8:58 AM
# @File:threading_demo1
import threading
import time


class TestThread(threading.Thread):
    def __init__(self, target=None, args=()):
        super(TestThread, self).__init__()
        self.target = target
        self.args = args

    def run(self):
        self.result = self.target(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except Exception:
            return None

def func1(sleep,sharearg):
    time.sleep(sleep)
    ct = time.time()
    print("func1结束{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))
    return [{"name": "func1","sharearg":sharearg}]

def func2(sleep,sharearg):
    time.sleep(sleep)
    ct = time.time()
    print("func2结束{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))
    return [{"name": "func2","sharearg":sharearg}]

def func3(sleep,sharearg):
    time.sleep(sleep)
    ct = time.time()
    print("func3结束{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))
    return [{"name": "func3","sharearg":sharearg}]

if __name__ == '__main__':
    start = time.time()
    results = []
    sharearg = "sharearg" # 使用多进程跑的时候不应该有共享变量或者共享类不然会出现等待，

    ct = time.time()
    print("开始{}.{}".format(time.strftime("%Y-%m-%d,%H:%M:%S", time.localtime(ct)), (ct - int(ct)) * 1000))
    # 创建线程
    thread_func1 = TestThread(target=func1,args=(0.4,sharearg,))
    thread_func2 = TestThread(target=func2,args=(0.2,sharearg,))
    # 启动线程
    thread_func1.start()
    thread_func2.start()

    results.extend(func3(0.9,sharearg,))

    thread_func1.join()
    thread_func2.join()

    results.extend(thread_func1.get_result())
    results.extend(thread_func2.get_result())

    print('Main thread has ended!',time.time()-start,results)


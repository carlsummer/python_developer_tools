# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/30/2021 8:51 AM
# @File:threading_utils
import threading


class ProcessThread1(threading.Thread):
    def __init__(self, idx,out_img):
        threading.Thread.__init__(self)
        self.idx = idx
        self.out_img=out_img

    def preprocess_image(self):
        self.out_img[self.idx] = "sdf" + str(self.idx)

    def run(self):
        self.preprocess_image()

if __name__ == '__main__':
    images = []

    threads = []
    for i in range(14):
        images.append(i)
        thread = ProcessThread1(i, images)
        thread.start()
        threads.append(thread)

    for thr in threads:
        thr.join()
    print(images)
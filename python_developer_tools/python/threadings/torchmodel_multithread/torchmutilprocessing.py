# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/30/2021 4:34 PM
# @File:torchmutilprocessing
import torch


def pianchuancuoweiResult(bigCut, queue):
    pianchuancuoweiResultList = []
    queue.put(pianchuancuoweiResultList)

queue = torch.multiprocessing.Queue()  # 获取返回结果的队列
thread_pcc = torch.multiprocessing.Process(target=pianchuancuoweiResult, args=(bigCutAlg, queue))
thread_pcc.start()
thread_pcc.join()
rstList.extend(queue.get())
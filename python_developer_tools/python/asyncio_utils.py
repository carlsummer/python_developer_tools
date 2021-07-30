# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/30/2021 8:46 AM
# @File:asyncio_utils
import asyncio

# 调用的方法要加async

# self.loop = asyncio.get_event_loop() There is no current event loop in thread 'Thread-7'用下面两句替换
self.loop = asyncio.new_event_loop()
asyncio.set_event_loop(self.loop)

creepageDistance = CreepageDistance(self.creepageDistanceModel)
creepageD_task = self.loop.create_task(creepageDistance.run(bigCutAlg))
self.loop.run_until_complete(creepageD_task)

rstList.extend(creepageD_task.result())
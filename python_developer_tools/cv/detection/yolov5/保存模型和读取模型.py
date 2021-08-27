# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/27/2021 9:47 AM
# @File:保存模型和读取模型
import os
import torch
import sys
# 保存模型
torch.save(ckpt, last_path)

# 读取模型
pwd = os.path.abspath(__file__)
# 当前文件的父路径
repo_dir = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")  # "/home/deploy/PVDefectPlatform/ztpanels/extra_apps/PVDefectAlgDeploy/scripts/"
sys.path.insert(0, repo_dir)
model = torch.load(weight_path, map_location=self.device)['model'].float().fuse().eval()  # load FP32 model torch.load 会去找models文件夹里面的
self.model = model.to(self.device)
sys.path.remove(repo_dir)
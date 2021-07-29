# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:03 AM
# @File:torch2onnx-integrate.py
import sys

import onnx
import torchvision

sys.path.insert(0, '/ours_code/hrnet/lib/')
sys.path.insert(1, '/ours_code/hrnet/')
sys.path.insert(1, '/')
import argparse
import base64
import glob
import json
import logging
import os
import pprint
import shutil
import sys
import time
import timeit
import asyncio

import albumentations as A
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from ai_hub import inferServer
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.nn import functional as F

from ours_code.hrnet.lib.models.seg_hrnet import get_seg_model
from ours_code.hrnet.lib.config import update_config, config

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation network")

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="/ours_code/hrnet/experiments/tc/seg_hrnet_w18_256x256_sgd_lr7e-3_wd1e-4_bs_16_epoch300.yaml",
                        type=str)
    parser.add_argument("--ckpt_path", help="checkpoint path",
                        default="/ours_code/hrnet/output/tc/best.pth", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    update_config(config, args)

    return args

class mymodel(nn.Module):

    def getmodel(self,ckpt_path):
        model = get_seg_model(config)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(ckpt_path)
        pretrained_dict = {
            k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict.keys()
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        model = model.cuda()
        model.eval()
        return model

    def __init__(self):
        super(mymodel, self).__init__()
        self.model1 = self.getmodel(r"/user_data/model_data/net3-train1/best.pth")
        self.model2 = self.getmodel(r"/user_data/model_data/net3-train2-2/best.pth")
        self.model3 = self.getmodel(r"/user_data/model_data/net3-train3-1/best.pth")

        # mean std
        self.mean = [0.485, 0.456, 0.406, 0.5]
        self.std = [0.229, 0.224, 0.225, 0.25]
        self.mean = torch.tensor(self.mean, dtype=torch.float32)
        self.mean.mul_(255.0)
        self.mean = self.mean.view(-1, 1, 1)
        self.std = torch.tensor(self.std, dtype=torch.float32)
        self.std.mul_(255.0)
        self.std = torch.reciprocal(self.std)
        self.std = self.std.view(-1, 1, 1)

        # asyncio
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def torchvisiontransformsNormalize(self,input2):
        torch.set_printoptions(precision=9)
        input2 = input2.permute(0,3, 1, 2)
        dtype = input2.dtype

        mean = torch.as_tensor(self.mean, dtype=dtype, device=input2.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=input2.device)

        input2.sub_(mean).mul_(std)

        return input2

    async def getpred(self,x,model):
        x = model(x)
        pred = F.interpolate(
            input=x,
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        )
        pred = pred.squeeze()
        pred = torch.argmax(pred, 0)
        return pred

    def forward(self, x):
        x = self.torchvisiontransformsNormalize(x)

        pred1_task = self.loop.create_task(self.getpred(x,self.model1))
        self.loop.run_until_complete(pred1_task)
        pred2_task = self.loop.create_task(self.getpred(x,self.model2))
        self.loop.run_until_complete(pred2_task)
        pred3_task = self.loop.create_task(self.getpred(x,self.model3))
        self.loop.run_until_complete(pred3_task)

        pred1=pred1_task.result()
        pred2=pred2_task.result()
        pred3=pred3_task.result()
        return pred1,pred2,pred3



if __name__ == "__main__":
    args = parse_args()
    mymodel = mymodel()

    img_path = "/user_data/tmp_data/tc/val_images/1_015401.tif"
    fin = open(img_path, 'rb')
    img = fin.read()
    bast64_data = base64.b64encode(img)
    bast64_str = str(bast64_data,'utf-8')
    img = bast64_str
    bast64_data = img.encode(encoding="utf-8")
    img = base64.b64decode(bast64_data)
    img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), -1) # 256*256*4
    dummy_input = torch.tensor(img).float().cuda()
    dummy_input = dummy_input.unsqueeze(0)
    # dummy_input = torch.randn(1,256, 256, 4).cuda()
    onnx_path = '/user_data/model_data/checkpoint-best.onnx'
    torch.onnx.export(mymodel, dummy_input, onnx_path, opset_version=11, verbose=True)

    # 检查导出的model
    onnx.checker.check_model(onnx.load(onnx_path))

    #mean_iu:0.6798058098508942,Fe:0.3554028411915428,final_score:0.5824849192530888
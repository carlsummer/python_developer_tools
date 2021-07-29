# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:04 AM
# @File:torch2onnx3-tta.py
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
import ttach
from ttach.base import Merger, Compose

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
    def __init__(self, ckpt_path):
        super(mymodel, self).__init__()
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

        self.model = model

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

        # tta
        self.transforms=Compose(
                [
                    ttach.HorizontalFlip(),
                    ttach.VerticalFlip(),
                ]
            )
        self.m1=Merger(type="tsharpen",n=len(self.transforms))


    def torchvisiontransformsNormalize(self,input2):
        torch.set_printoptions(precision=9)
        input2 = input2.permute(0,3, 1, 2)
        dtype = input2.dtype

        mean = torch.as_tensor(self.mean, dtype=dtype, device=input2.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=input2.device)

        input2.sub_(mean).mul_(std)

        return input2

    def forward(self, x):
        # x = x.squeeze()
        x = self.torchvisiontransformsNormalize(x)
        # x = x.unsqueeze(0)  # 转为1,4,256,256
        datas=[]
        for transformer in self.transforms:
            augmented_image=transformer.augment_image(x)
            datas.append(augmented_image)
        datas=torch.cat(datas,0)
        pre_batch=self.model(datas)
        for index,transformer in enumerate(self.transforms):
            pre = transformer.deaugment_mask(pre_batch[index].unsqueeze(0))
            self.m1.append(pre)
        pre1=self.m1.result

        pred = F.interpolate(
            input=pre1,
            size=(256,256),
            mode="bilinear",
            align_corners=False,
        )
        pred=pred.squeeze()
        pred = torch.argmax(pred, 0) + 1
        return pred



if __name__ == "__main__":
    args = parse_args()
    mymodel = mymodel(args.ckpt_path)

    # img_path = "/user_data/tmp_data/tc/val_images/1_015401.tif"
    # fin = open(img_path, 'rb')
    # img = fin.read()
    # bast64_data = base64.b64encode(img)
    # bast64_str = str(bast64_data,'utf-8')
    # img = bast64_str
    # bast64_data = img.encode(encoding="utf-8")
    # img = base64.b64decode(bast64_data)
    # img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), -1) # 256*256*4
    # dummy_input = torch.tensor(img).float().cuda()
    # dummy_input = dummy_input.unsqueeze(0)
    dummy_input = torch.randn(1,256, 256, 4).cuda()
    onnx_path = '/user_data/model_data/checkpoint-best.onnx'
    torch.onnx.export(mymodel, dummy_input, onnx_path, opset_version=11, verbose=True)

    # 检查导出的model
    onnx.checker.check_model(onnx.load(onnx_path))


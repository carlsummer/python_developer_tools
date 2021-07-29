# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:04 AM
# @File:torch2onnx2.py
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
        self.transform = torchvision.transforms.Compose([
            # 标准化处理-->转换为标准正太分布，使模型更容易收敛
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406, 0.5],
                std=[0.229, 0.224, 0.225, 0.25]
            )
        ])

    def forward(self, x):
        # x = x.transpose((2, 0, 1))
        # x = self.transform(x)
        # x=x.unsqueeze(0) # 转为1,4,256,256
        x = self.model(x)
        pred = F.interpolate(
            input=x,
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

    dummy_input = torch.randn(1,4,256,256).cuda()
    onnx_path = '/user_data/model_data/checkpoint-best.onnx'
    torch.onnx.export(mymodel, dummy_input, onnx_path, opset_version=11, verbose=True)

    # 检查导出的model
    onnx.checker.check_model(onnx.load(onnx_path))


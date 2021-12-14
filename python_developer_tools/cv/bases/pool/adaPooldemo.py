# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:12/14/2021 2:31 PM
# @File:adaPool
# Ensure that you import `torch` first!
import torch
import adapool_cuda

# For function calls
from adaPool import adapool1d, adapool2d, adapool3d, adaunpool
from adaPool import edscwpool1d, edscwpool2d, edscwpool3d
from adaPool import empool1d, empool2d, empool3d
from adaPool import idwpool1d, idwpool2d, idwpool3d

# For class calls
from adaPool import AdaPool1d, AdaPool2d, AdaPool3d
from adaPool import EDSCWPool1d, EDSCWPool2d, EDSCWPool3d
from adaPool import EMPool1d, EMPool2d, EMPool3d
from adaPool import IDWPool1d, IDWPool2d, IDWPool3d

import numpy as np
import cv2

from collections import OrderedDict
import glob

beta = (128,128)
delta = (256,256)

beta_ones = torch.ones((128,128), device='cuda:0')
beta_zeros = torch.zeros((128,128), device='cuda:0')

images = []

# 这个for 可以注释掉
for img in glob.glob('/home/zengxh/workspace/python_developer_tools/docs/whls/adaPool-main/images/original/*.png'):
    image = cv2.imread(img)
    image = cv2.resize(image, (256,256))
    cv2.imwrite('unpool/'+img.split('.')[0].split('/')[1]+'_original.jpg',image)
    image = image.astype(np.float32)
    image /= 255.
    images = torch.as_tensor(image).permute(2,0,1).unsqueeze(0).cuda(non_blocking=True)

    pool = {'adapool': AdaPool2d(kernel_size=2, stride=2, beta=beta, dtype=images.dtype, device=images.get_device(), return_mask=True),
            'empool': AdaPool2d(kernel_size=2, stride=2, beta=beta_zeros, dtype=images.dtype, device=images.get_device(), return_mask=True),
            'eidwpool': AdaPool2d(kernel_size=2, stride=2, beta=beta_ones, dtype=images.dtype, device=images.get_device(), return_mask=True),
            'idwpool': IDWPool2d(kernel_size=2, stride=2) }


    for key in pool.keys():
        print(img,key)
        if key!='idwpool':
            tmp, mask = pool[key](images)
            un_tmp = adaunpool(x=tmp, mask=mask)
            un_img_ = un_tmp.detach().clone().to('cpu').squeeze(0).permute(1,2,0).numpy() * 255.
            img_ = tmp.detach().clone().to('cpu').squeeze(0).permute(1,2,0).numpy() * 255.
            cv2.imwrite('generated/'+img.split('.')[0].split('/')[1]+'_{}.jpg'.format(key),img_)
            cv2.imwrite('unpool/'+img.split('.')[0].split('/')[1]+'_{}.jpg'.format(key),un_img_)
        else:
            tmp= pool[key](images)
            img_ = tmp.cpu().detach().squeeze(0).permute(1,2,0).numpy() * 255.
            cv2.imwrite('generated/'+img.split('.')[0].split('/')[1]+'_{}.jpg'.format(key),img_)

images = torch.rand((32, 3, 256, 256)).cuda()
beta = (64,64)
delta = (128,128)

model = torch.nn.Sequential(OrderedDict([
    ('conv1' , torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)),
    ('bn1' , torch.nn.BatchNorm2d(64)),
    ('relu' , torch.nn.ReLU(inplace=True)),
    ('pool'  , AdaPool2d(kernel_size=2, stride=2, beta=beta,  dtype=images.dtype, device=images.get_device())),
    ('conv2' , torch.nn.Conv2d(kernel_size=3, stride=1, in_channels=64, out_channels=64))
    ]))

model = model.cuda()
model.train()

print('named params:')
for n, p in model.named_parameters():
    print(n)


for i in range(100):
    torch.cuda.empty_cache()
    images = torch.rand((32, 3, 256, 256), device='cuda:0')
    tmp = model(images)
    tmp.mean().pow(2).backward()
    print('Successfully completed iteration {} \n'.format(i))
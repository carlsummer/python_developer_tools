# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/5/2021 3:07 PM
# @File:swaoffline
import glob

import torch
import torch.nn as nn
from tqdm import tqdm

def process(param):
    new_pre = {}
    for k,v in param.items():
        name = k[7:]
        # new_pre[name]=v
        new_pre[k] = v
    return new_pre

def swa_offline(weight_path_list):
    merge_weight = {}
    net_weight_list = []
    num = len(weight_path_list)
    for i in tqdm(range(num)):
        #如果是多卡训练的，这里需要调用process函数先处理下权重，单卡的不需要
        # net_weight_list.append(process(torch.load(weight_path_list[i])['state_dict']))
        net_weight_list.append(torch.load(weight_path_list[i])['model_state_dict'])
    for key in net_weight_list[0].keys():
        tmp_list = [per[key] for per in net_weight_list]
        tmp = torch.true_divide(tmp_list[0], num)  # tmp_list[0]/num
        for j in range(1, num):
            tmp += torch.true_divide(tmp_list[j],num)  # tmp_list[j]/num
        merge_weight[key] = tmp
    return merge_weight

if __name__ == '__main__':
    # path_1='/home/zengxh/workspace/lcnn/logs/210806-012458-88f281a-baseline/npz/000036000/checkpoint.pth'
    # path_2='/home/zengxh/workspace/lcnn/logs/210806-012458-88f281a-baseline/npz/000035250/checkpoint.pth'
    # path_3='/home/zengxh/workspace/lcnn/logs/210806-012458-88f281a-baseline/npz/000034500/checkpoint.pth'
    # path_4='/home/zengxh/workspace/lcnn/logs/210806-012458-88f281a-baseline/npz/000033750/checkpoint.pth'
    # path_5='/home/zengxh/workspace/lcnn/logs/210806-012458-88f281a-baseline/npz/000033000/checkpoint.pth'
    # path_6='/home/zengxh/workspace/lcnn/logs/210806-012458-88f281a-baseline/npz/000032250/checkpoint.pth'

    model_list=glob.glob('/home/zengxh/workspace/lcnn/logs/210806-012458-88f281a-baseline/npz/*/*.pth')
    # model_list = [path_1,path_2,path_3,path_4,path_5,path_6]
    print(model_list)

    merge_weight=swa_offline(model_list[20:])
    torch.save(
        {
            "arch": merge_weight.__class__.__name__,
            "model_state_dict": merge_weight,
        },
        'swa_b6.pth',
    )
    #model.load_state_dict(merge_weight)
    print("finished ..........")
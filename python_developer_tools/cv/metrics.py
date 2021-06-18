# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:6/4/2021 8:41 AM
# @File:metrics
import torch
def get_distance_acc_tensor(outputs, targets,px=64):
    """获取距离准确率"""
    acc_list = torch.tensor([]).to(outputs.device)
    for pred_sub in torch.abs(torch.sub(targets*64, outputs*64)):
        if pred_sub < px:
            acc = 1 - torch.mul((1 / px), pred_sub)
            acc_list = torch.cat((acc_list, acc.unsqueeze(0)), 0)
    return torch.div(acc_list.sum(),len(outputs))
    # return torch.mean(1 - torch.mul((1 / px), torch.abs(torch.sub(targets*64, outputs*64))))

def get_distance_acc(l1loss,px=32):
    """获取距离准确率"""
    return 1-(1/px)*l1loss if l1loss<px else 0
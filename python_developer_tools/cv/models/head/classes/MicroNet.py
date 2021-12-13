# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:12/13/2021 2:50 PM
# @File:MicroNet
import torch
import torch.nn as nn

from python_developer_tools.cv.bases.FC.SwishLinear import SwishLinear


def MicroNet_head(input_channel,output_channel,dropout_rate,num_classes):
    return nn.Sequential(
            SwishLinear(input_channel, output_channel),
            nn.Dropout(dropout_rate),
            SwishLinear(output_channel, num_classes)
        )



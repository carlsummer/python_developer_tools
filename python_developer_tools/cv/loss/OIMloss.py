# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/30/2021 9:07 AM
# @File:OIMloss
import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, lut, momentum=0.5):
        ctx.lut = lut  # torch.Size([625, 128])
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)  # inputs: torch.Size([64, 128])
        outputs = inputs.mm(ctx.lut.t())  # (64, 128) * (128, 625)
        return outputs  # torch.Size([64, 625])

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.lut)
        for x, y in zip(inputs, targets):
            ctx.lut[y] = ctx.momentum * ctx.lut[y] + (1. - ctx.momentum) * x
            ctx.lut[y] /= ctx.lut[y].norm()
        return grad_inputs, None, None, None, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM.apply(inputs, targets, lut, torch.Tensor([momentum]).to(inputs.device))  # momentum=momentum


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features  # 512
        self.num_classes = num_classes  # 625
        self.momentum = momentum  # 0.5
        self.scalar = scalar  # 30
        self.weight = weight  # None
        self.register_buffer('lut', torch.zeros(num_classes, num_features))
        self.size_average = size_average  # True

    def forward(self, inputs, targets):  #
        # batchsize = inputs.size(0)
        # seq_len = inputs.size(1)
        # inputs = inputs.view(batchsize*seq_len, -1)
        # targets = targets.view(batchsize*seq_len)
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        # inputs = score.expand_as(inputs) * inputs
        inputs *= self.scalar

        loss = F.cross_entropy(inputs, targets, weight=self.weight)
        return loss, inputs, self.lut




# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:9/1/2021 8:26 AM
# @File:topk_crossEntrophy
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


class topk_crossEntrophy(nn.Module):
    def __init__(self, top_k=0.7,device="cuda:0"):
        super(topk_crossEntrophy, self).__init__()
        self.loss = nn.NLLLoss()
        self.top_k = top_k
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device

    def forward(self, input, target):
        softmax_result = self.softmax(input)

        loss = Variable(torch.zeros(1,1)).to(self.device)
        for idx, row in enumerate(softmax_result):
            gt = target[idx]
            gt = torch.unsqueeze(gt, 0)
            pred = torch.unsqueeze(row, 0)
            cost = self.loss(pred, gt)
            loss = torch.cat((loss, cost.view(1,1)), 0)

        loss = loss[1:]
        index = torch.topk(loss.view(1,-1), int(self.top_k * loss.size()[0]))
        valid_loss = index[0]  # loss[index[1]]

        return torch.mean(valid_loss)


if __name__ == '__main__':
    a = torch.randn((10, 2))
    a.normal_()
    b = np.random.randint(2, size=10)
    b = torch.from_numpy(b.astype(np.float32)).type(torch.LongTensor)

    topk_loss = topk_crossEntrophy()
    loss = topk_loss(Variable(a, requires_grad=True), Variable(b))
    print(loss.detach().numpy())

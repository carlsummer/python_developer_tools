# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/11/2021 2:27 PM
# @File:SSM
import torch
import torch.nn as nn

class SSM(nn.Module):
    def __init__(self,num_channels=2048, out_channels=1000,num_heads=4):
        super(SSM,self).__init__()
        self.num_heads = num_heads
        self.n = int(num_channels / self.num_heads)
        layers = []
        for i in range(self.num_heads):
            in_features = i * self.n
            out_features = (i + 1) * self.n
            if i == self.num_heads -1:
                ssm = nn.Sequential(
                    nn.Linear(out_features, out_channels)
                )
            else:
                ssm = nn.Sequential(
                    nn.BatchNorm1d(out_features),
                    nn.ReLU(inplace=True),
                    nn.Linear(out_features, out_channels)
                )
            layers.append(ssm)
        self.ssm_layers = nn.Sequential(*layers)
        print(self.ssm_layers)

    def forward(self,features):
        for i in range(self.num_heads):
            in_features = i * self.n
            out_features = (i + 1) * self.n
            x = features[:,:out_features]
            if i == 0:
                result = self.ssm_layers[i](x)
            else:
                result += self.ssm_layers[i](x)
        return result / self.num_heads

if __name__ == '__main__':
    x = torch.randn(2, 2048, 1, 1) # batch_size需要大于2
    x = x.view(x.size(0), -1)

    fc = nn.Linear(2048, 1000)
    out = fc(x)
    print(out.shape)

    model = SSM()
    out = model(x)
    print(out.shape)
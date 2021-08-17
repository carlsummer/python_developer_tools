# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/15/2021 3:24 PM
# @File:DynamicReLU
# https://github.com/Islanna/DynamicReLU
import torch
import torch.nn as nn


class DyReLUA(nn.Module):
    def __init__(self,
                 channels,
                 reduction=4,
                 k=2):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2 * k, 1),
            nn.Sigmoid()
        )

        # default parameter setting
        # lambdaA = 1.0, lambdaB = 0.5;
        # alphaA1 = 1, alphaA2=alphaB1=alphaB2=0
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def forward(self, x):
        coef = self.coef(x)
        coef = 2 * coef - 1
        coef = coef.view(-1, 2 * self.k) * self.lambdas + self.bias

        # activations
        # NCHW --> NCHW1
        x_perm = x.permute(1, 2, 3, 0).unsqueeze(-1)
        # HWNC1 * NK --> HWCNK
        output = x_perm * coef[:, :self.k] + coef[:, self.k:]
        result = torch.max(output, dim=-1)[0].permute(3, 0, 1, 2)
        return result


class DyReLUB(nn.Module):
    def __init__(self,
                 channels,
                 reduction=4,
                 k=2):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2 * k * channels, 1),
            nn.Sigmoid()
        )

        # default parameter setting
        # lambdaA = 1.0, lambdaB = 0.5;
        # alphaA1 = 1, alphaA2=alphaB1=alphaB2=0
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def forward(self, x):
        coef = self.coef(x)
        coef = 2 * coef - 1

        # coefficient update
        coef = coef.view(-1, self.channels, 2 * self.k) * self.lambdas + self.bias

        # activations
        # NCHW --> HWNC1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # HWNC1 * NCK --> HWNCK
        output = x_perm * coef[:, :, :self.k] + coef[:, :, self.k:]
        # maxout and HWNC --> NCHW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result


class DyReLUC(nn.Module):
    def __init__(self,
                 channels,
                 reduction=4,
                 k=2,
                 tau=10,
                 gamma=1 / 3):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.k = k
        self.tau = tau
        self.gamma = gamma

        self.coef = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, 2 * k * channels, 1),
            nn.Sigmoid()
        )
        self.sptial = nn.Conv2d(channels, 1, 1)

        # default parameter setting
        # lambdaA = 1.0, lambdaB = 0.5;
        # alphaA1 = 1, alphaA2=alphaB1=alphaB2=0
        self.register_buffer('lambdas', torch.Tensor([1.] * k + [0.5] * k).float())
        self.register_buffer('bias', torch.Tensor([1.] + [0.] * (2 * k - 1)).float())

    def forward(self, x):
        N, C, H, W = x.size()
        coef = self.coef(x)
        coef = 2 * coef - 1

        # coefficient update
        coef = coef.view(-1, self.channels, 2 * self.k) * self.lambdas + self.bias

        # spatial
        gamma = self.gamma * H * W
        spatial = self.sptial(x)
        spatial = spatial.view(N, self.channels, -1) / self.tau
        spatial = torch.softmax(spatial, dim=-1) * gamma
        spatial = torch.clamp(spatial, 0, 1).view(N, 1, H, W)

        # activations
        # NCHW --> HWNC1
        x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
        # HWNC1 * NCK --> HWNCK
        output = x_perm * coef[:, :, :self.k] + coef[:, :, self.k:]

        # permute spatial from NCHW to HWNC1
        spatial = spatial.permute(2, 3, 0, 1).unsqueeze(-1)
        output = spatial * output

        # maxout and HWNC --> NCHW
        result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)
        return result

def convert_relu_to_DyReLU(model,dyrelutype="A"):
    # 将relu替换为DyReLUA
    model_ft_modules = list(model.modules())
    dyReluchannels = []
    for i, (m, name) in enumerate(zip(model.modules(), model.named_modules())):
        if type(m) is nn.ReLU:
            t_layers = model_ft_modules[i - 1]
            if hasattr(t_layers, "num_features"):
                channels = t_layers.num_features
            if hasattr(t_layers, "out_channels"):
                channels = t_layers.out_channels
            if hasattr(t_layers, "out_features"):
                channels = t_layers.out_features

            if dyrelutype == "A":
                dyReluchannels.append({"name": name, "dyrelu": DyReLUA(channels)})
            if dyrelutype == "B":
                dyReluchannels.append({"name": name, "dyrelu": DyReLUB(channels)})
            if dyrelutype == "C":
                dyReluchannels.append({"name": name, "dyrelu": DyReLUC(channels)})
            del channels
    for dictsss in dyReluchannels:
        setattr(model, dictsss["name"][0], dictsss["dyrelu"])
    return model

if __name__ == '__main__':
    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv1 = nn.Conv2d(3, 10, 5)
            self.relu = DyReLUA(10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            return x
    input = torch.randn(2,3,114,114)
    model = Model()
    out = model(input)
    print(out.shape)
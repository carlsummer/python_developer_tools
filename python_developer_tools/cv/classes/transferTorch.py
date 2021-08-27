"""
迁移学习的网络结构
"""
import os

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F

try:
    import softpool_cuda
    from SoftPool import soft_pool2d, SoftPool2d
except ImportError:
    print('Please install SoftPool first: https://github.com/alexandrosstergiou/SoftPool')

def Model(model_name, nc, pretrained):
    """获取网络模型"""
    return eval(model_name)(nc, pretrained)


def shufflenet_v2_x0_5(nc, pretrained):
    model_ft = models.shufflenet_v2_x0_5(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft

def shufflenet_v2_x1_0(nc, pretrained):
    model_ft = models.shufflenet_v2_x1_0(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft

def shufflenet_v2_x1_5(nc, pretrained):
    model_ft = models.shufflenet_v2_x1_5(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft

def shufflenet_v2_x2_0(nc, pretrained):
    model_ft = models.shufflenet_v2_x2_0(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft


def squeezenet1_1(nc, pretrained):
    model_ft = models.squeezenet1_1(pretrained=pretrained)
    model_ft.classifier[1] = nn.Conv2d(512, nc, kernel_size=1)
    return model_ft


def mobilenet_v2(nc, pretrained):
    """pretrained 为true，那么会加载imageNet训练好的"""
    model_ft = models.mobilenet_v2(pretrained=pretrained)
    model_ft.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model_ft.last_channel, nc),
    )
    return model_ft

def inception_v3(nc, pretrained):
    model_ft = models.inception_v3(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft


def getVggNet(nc):
    model_ft = models.vgg11(pretrained=False)
    # 对迁移模型进行调整
    for parma in model_ft.parameters():
        parma.requires_grad = False

    model_ft.classifier = nn.Sequential(nn.Linear(25088, nc))
    return model_ft


def resnet18(nc, pretrained):
    model_ft = models.resnet18(pretrained=pretrained)

    # 使用softpool
    # nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 可以用下面替换
    # model_ft.maxpool = SoftPool2d(kernel_size=(2,2), stride=(2,2))

    # 可以用的情况2
    # self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
    # self.pool2 = SoftPool2d(kernel_size=3, stride=2)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft


def resnet152(nc, pretrained):
    model_ft = models.resnet152(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft


def getmnasnet1_0(nc):
    model_ft = models.mnasnet1_0(pretrained=True)
    model_ft.classifier = nn.Sequential(
        # nn.Linear(1280, 1280),
        nn.Linear(1280, nc),
    )
    return model_ft

if __name__ == '__main__':
    model = inception_v3(2,False)
    dummy_input = torch.randn(1, 3, 640, 640)
    onnx_path = os.path.join("netron_model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path)
# python_developer_tools
> python 开发过程中常用到的工具;包括网站开发,人工智能,文件，数据类型转换
> 支付接口对接，外挂，bat,sh等我在工作中常用到的开发工具

## 安装和使用
```shell script
pip uninstall python_developer_tools
pip install git+https://github.com/carlsummer/python_developer_tools.git
from python_developer_tools import cv
```

***
# Contents
- [cv 计算机视觉](#cv-计算机视觉)
    - [基础组成成分](#基础组成成分)
        - [Convolution Series](#Convolution-series)
            - [Depthwise Separable Convolution Usage](#Depthwise-Separable-Convolution-Usage)
    - [分类模型classnetwork]
        - [AlexNet](#AlexNet)
        - [DenseNet](#DenseNet)
        - [Efficientnet](#Efficientnet)
        - [InceptionV1](#InceptionV1)
        - [InceptionV2](#InceptionV2)
        - [InceptionV3](#InceptionV3)
        - [repVGGNet](#repVGGNet)
        - [ResNet](#ResNet)
        - [ResNeXt](#ResNeXt)
        - [VGGNet](#VGGNet)
        - [GhostNet](#GhostNet)
        - [MixNet](#MixNet)
        - [MobileNetV1](#MobileNetV1)
        - [MobileNetV2](#MobileNetV2)
        - [MobileNetV3](#MobileNetV3)
        - [MobileNetXt](#MobileNetXt)
        - [ShuffleNet](#ShuffleNet)
        - [ShuffleNetV2](#ShuffleNetV2)
        - [SqueezeNet](#SqueezeNet)
        - [Xception](#Xception)
- [files](#files)
    - [common](#常用)
        - [get_filename_suf_pix](#get_filename_suf_pix)
***

***
#### Depthwise Separable Convolution Usage
##### Paper
["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/abs/1704.04861)

##### Overview
![](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/img/DepthwiseSeparableConv.png)

##### Code
感谢[代码来源External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch#1-Depthwise-Separable-Convolution-Usage)
```python
import torch
from python_developer_tools.cv.bases.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
input=torch.randn(1,3,224,224)
dsconv=DepthwiseSeparableConvolution(3,64)
out=dsconv(input)
```
***

***
### AlexNet
##### Code
```python
import torch
from python_developer_tools.cv.classes.AlexNet import AlexNet
model = AlexNet()
input = torch.randn(8,3,224,224)
out = model(input)
print(out.shape)
```
***
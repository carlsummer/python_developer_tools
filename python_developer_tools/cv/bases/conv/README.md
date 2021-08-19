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
#### MBConv
##### Paper
[Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a.html)

##### Overview
![](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/img/MBConv.jpg)

##### Code
感谢[代码来源External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch#1-Depthwise-Separable-Convolution-Usage)
```python
from python_developer_tools.cv.bases.conv.MBConv import MBConvBlock
import torch
input=torch.randn(1,3,112,112)
mbconv=MBConvBlock(ksize=3,input_filters=3,output_filters=3,image_size=112)
out=mbconv(input)
print(out.shape)
```
***

***
#### Involution
##### Paper
[Inverting the Inherence of Convolution for Visual Recognition](https://arxiv.org/abs/2103.06255)

##### Overview
![](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/img/Involution.png)

##### Code
感谢[代码来源External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch#1-Depthwise-Separable-Convolution-Usage)
```python
from python_developer_tools.cv.bases.conv.Involution import Involution
import torch
input=torch.randn(1,4,64,64)
involution=Involution(kernel_size=3,in_channel=4,stride=2)
out=involution(input)
print(out.shape)
```
***

## 一文看尽深度学习中的20种卷积（附源码整理和论文解读）
> https://mp.weixin.qq.com/s/B1IDfCG4WqHYxsPkoVKEjg

【经典卷积系列】
原始卷积 (Vanilla Convolution)
组卷积 (Group convolution)
转置卷积 (Transposed Convolution)
1×1卷积 (1×1 Convolution)
空洞卷积 (Atrous convolution)
可变形卷积 (Deformable convolution)
空间可分离卷积 (Spatially Separable Convolution)
图卷积 (Graph Convolution)
植入块 (Inception Block)
【卷积变体系列】
非对称卷积(Asymmetric Convolution)
八度卷积(Octave Convolution)
异构卷积(Heterogeneous Convolution)
条件参数化卷积(Conditionally Parameterized Convolutions)
动态卷积(Dynamic Convolution)
幻影卷积(Ghost Convolution)
自校正卷积(Self-Calibrated Convolution)

## DO-Conv 逐深度过参数化卷积(Depthwise Over-parameterized Convolution)
1. [github代码](https://github.com/yangyanli/DO-Conv)
2. [论文](https://arxiv.org/pdf/2006.12030.pdf)

分离注意力模块(ResNeSt Block)

## res2net

## CoConv
1. [github代码](https://github.com/iduta/coconv)
2. [论文](https://arxiv.org/abs/2108.07387)

<div align="center">
<img src="https://img.shields.io/badge/-Python-brightgreen">
<img src="https://img.shields.io/badge/-%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90-yellowgreen">
<img src="https://img.shields.io/badge/-%E7%AE%97%E6%B3%95-yellow">
<img src="https://img.shields.io/badge/-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-lightgrey">
</div>

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
            - [MBConv](#MBConv)
            - [Involution](#Involution)
        - [全连接FC]
            - [SSM](#SSM)
    - 分类classes
        - demo
            - [训练cifar10数据集](./python_developer_tools/cv/classes/demo/train_cifar10.py)
    - train
        - 二阶段训练
            - [swa](./python_developer_tools/cv/train/二阶段训练/swa_pytorch.py)
    - utils
        - [tensorboard](./python_developer_tools/cv/utils/tensorboard_demo.py)
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
    - [common](#common)
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

#### SSM
##### Paper
[Exploiting Featureswith Split-and-Share Module](https://arxiv.org/abs/2108.04500)
##### Overview
![](./temimg/SSM.png)
##### code
```python
from python_developer_tools.cv.bases.FC.SSM import SSM
import torch
x = torch.randn(2, 2048, 1, 1)
x = x.view(x.size(0), -1)
model = SSM()
out = model(x)
print(out.shape)
```

***
### AlexNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.AlexNet import AlexNet
model = AlexNet()
input = torch.randn(8,3,224,224)
out = model(input)
print(out.shape)
```
***

***
### DenseNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.DenseNet import DenseNet121
model = DenseNet121()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### Efficientnet
##### Code
```python
import torch
from python_developer_tools.cv.classes.Efficientnet import EfficientNet
model = EfficientNet('efficientnet_b0')
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### InceptionV1
#### Code
```python
import torch
from python_developer_tools.cv.classes.InceptionV1 import InceptionV1
model = InceptionV1()
input = torch.randn(1, 3, 224, 224)
aux1, aux2, out = model(input)
print(aux1.shape)
print(aux2.shape)
print(out.shape)
```
***

***
### InceptionV2
#### Code
```python
import torch
from python_developer_tools.cv.classes.InceptionV2 import InceptionV2
model = InceptionV2()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### InceptionV3
#### Code
```python
import torch
from python_developer_tools.cv.classes.InceptionV3 import InceptionV3
model = InceptionV3()
input = torch.randn(1, 3, 299, 299)
aux,out = model(input)
print(aux.shape)
print(out.shape)
```
***

***
### repVGGNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.repVGGNet import RepVGG_A1
model = RepVGG_A1()
input = torch.randn(1,3,224,224)
out = model(input)
print(out.shape)
```
***

***
### ResNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.ResNet import ResNet50
model = ResNet50()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### ResNeXt
#### Code
```python
import torch
from python_developer_tools.cv.classes.ResNeXt import ResNeXtBlock
model = ResNeXtBlock(in_places=256, places=128)
input = torch.randn(1,256,64,64)
out = model(input)
print(out.shape)
```
***

***
### VGGNet
##### Code
```python
import torch
from python_developer_tools.cv.classes.VGGNet import VGG16
model = VGG16()
input = torch.randn(1,3,224,224)
out = model(input)
print(out.shape)
```
***

***
### GhostNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.GhostNet import GhostNet
model = GhostNet()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### MixNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.MixNet import MixNet
model = MixNet(type ='mixnet_m')
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### MobileNetV1
#### Code
```python
import torch
from python_developer_tools.cv.classes.MobileNetV1 import MobileNetV1
model = MobileNetV1()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### MobileNetV2
#### Code
```python
import torch
from python_developer_tools.cv.classes.MobileNetV2 import MobileNetV2
model = MobileNetV2()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### MobileNetV3
#### Code
```python
import torch
from python_developer_tools.cv.classes.MobileNetV3 import MobileNetV3
model = MobileNetV3(type='small')
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### MobileNetXt
#### Code
```python
import torch
from python_developer_tools.cv.classes.MobileNetXt import MobileNetXt
model = MobileNetXt()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### ShuffleNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.ShuffleNet import shufflenet_g1
model = shufflenet_g1()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### ShuffleNetV2
##### Code
```python
import torch
from python_developer_tools.cv.classes.ShuffleNetV2 import shufflenet_v2_x2_0
model = shufflenet_v2_x2_0()
input = torch.randn(1, 3, 224, 224)
out = model(input)
print(out.shape)
```
***

***
### SqueezeNet
#### Code
```python
import torch
from python_developer_tools.cv.classes.SqueezeNet import SqueezeNet
model = SqueezeNet()
input = torch.rand(1,3,224,224)
out = model(input)
print(out.shape)
```
***

***
### Xception
#### Code
```python
import torch
from python_developer_tools.cv.classes.Xception import Xception
model = Xception()
input = torch.randn(1,3,299,299)
output = model(input)
print(output.shape)
```
***

***
# files
## common
<table>
    <thead>
        <tr><th>名称</th><th>功能</th></tr>
    </thead>
    <tbody>
        <tr><td>get_filename_suf_pix</td><td>获取路径的文件名,后缀,父路径</td></tr>
    </tbody>
</table>
***
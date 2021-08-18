<div align="center">
<img src="https://img.shields.io/badge/-Python-brightgreen">
<img src="https://img.shields.io/badge/-%E6%95%B0%E6%8D%AE%E5%88%86%E6%9E%90-yellowgreen">
<img src="https://img.shields.io/badge/-%E7%AE%97%E6%B3%95-yellow">
<img src="https://img.shields.io/badge/-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-lightgrey">
</div>

# python_developer_tools
> python 开发过程中常用到的工具;包括网站开发,人工智能,文件，数据类型转换
> 支付接口对接，外挂，bat,sh等我在工作中常用到的开发工具
> 1. 制作一个可插拔的python开发工具
> 2. 论文复现
> 3. 深度学习tricks收集

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
            - **Depthwise Separable Convolution Usage**
            - **MBConv**
            - **Involution**
        - [attentions注意力机制](./python_developer_tools/cv/bases/attentions/README.md)
            - **Squeeze-and-Excitation Networks**
        - [activates 激活函数](#activates)
            - **Adaptively-Parametric-ReLU**
            - **DynamicReLU**
        - [全连接FC]
            - [SSM](#SSM)
    - 分类classes
        - demo
            - [训练cifar10数据集](test/train_cifar10.py)
        - [分类模型classnetwork](#classnetwork)
    - datasets
        - [数据增强](./python_developer_tools/cv/datasets/README.md)
            - 分类任务数据增强
                - 图片自动对比度
                - 直方图增强
            - 直线检测数据增强
                - **cutout**
                - 旋转透视变换
        - 数据集读取
    - [scheduler](./python_developer_tools/cv/scheduler/README.md)
        - **ExpLR**
        - **WarmupExponentialLR**
        - **StepLR**
        - **WarmupStepLR**
        - **MultiStepLR**
        - **WarmupMultiStepLR**
        - **CosineLR**
        - **WarmupCosineAnnealingLR**
        - **LambdaLR**
        - **ReduceLROnPlateau**
        - **CosineAnnealingWarmRestarts**
        - **CyclicLR**
        - **OneCycleLR**
        - **PolyLR**
    - [optimizer](./python_developer_tools/cv/optimizer/README.md)
        - **SGD**
        - **ASGD**
        - **Adagrad**
        - **Adadelta**
        - **RMSprop**
        - **Adam**
        - **Adamax**
        - **SparseAdam**
        - **L-BFGS**
        - **Rprop**
        - **AdamW**
        - **RAdam**
        - **Ranger**
    - train
        - 二阶段训练
            - [swa](./python_developer_tools/cv/train/二阶段训练/swa_pytorch.py)
    - utils
        - [tensorboard](./python_developer_tools/cv/utils/tensorboard_demo.py)
    
- [files](#files)
    - [common](#common)
    - [pickle](#pickle)
- [python]
    - [threadings](#threadings)
        - multiprocessing_utils
***

### [Convolution-series](./python_developer_tools/cv/bases/conv/README.md)
| 名称      |    csdn | 
| :-------- | --------:| 
| [Depthwise Separable Convolution](./python_developer_tools/cv/bases/conv/DepthwiseSeparableConvolution.py)  | |
| [MBConv](./python_developer_tools/cv/bases/conv/MBConv.py)  | |
| [Involution](./python_developer_tools/cv/bases/conv/Involution.py)  | |

### [activates](./python_developer_tools/cv/bases/activates/README.md)
| 名称      |    csdn | 
| :-------- | --------:| 
| [Adaptively-Parametric-ReLU](./python_developer_tools/cv/bases/activates/APReLU.py)  | |
| [DynamicReLU](./python_developer_tools/cv/bases/activates/DynamicReLU.py)  | [解析](https://blog.csdn.net/Carlsummer/article/details/119730645)|


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

### classnetwork
| 名称      |    代码实例 | 
| :-------- | --------:| 
| AlexNet  | [实现](./python_developer_tools/cv/classes/AlexNet.py)|
| DenseNet  | [实现](./python_developer_tools/cv/classes/DenseNet.py)|
| Efficientnet  | [实现](./python_developer_tools/cv/classes/Efficientnet.py)|
| InceptionV1  | [实现](./python_developer_tools/cv/classes/InceptionV1.py)|
| InceptionV2  | [实现](./python_developer_tools/cv/classes/InceptionV2.py)|
| InceptionV3  | [实现](./python_developer_tools/cv/classes/InceptionV3.py)|
| repVGGNet  | [实现](./python_developer_tools/cv/classes/repVGGNet.py)|
| ResNet  | [实现](./python_developer_tools/cv/classes/ResNet.py)|
| ResNeXt  | [实现](./python_developer_tools/cv/classes/ResNeXt.py)|
| VGGNet  | [实现](./python_developer_tools/cv/classes/VGGNet.py)|
| GhostNet  | [实现](./python_developer_tools/cv/classes/GhostNet.py)|
| MixNet  | [实现](./python_developer_tools/cv/classes/MixNet.py)|
| MobileNetV1  | [实现](./python_developer_tools/cv/classes/MobileNetV1.py)|
| MobileNetV2  | [实现](./python_developer_tools/cv/classes/MobileNetV2.py)|
| MobileNetV3  | [实现](./python_developer_tools/cv/classes/MobileNetV3.py)|
| MobileNetXt  | [实现](./python_developer_tools/cv/classes/MobileNetXt.py)|
| ShuffleNet  | [实现](./python_developer_tools/cv/classes/ShuffleNet.py)|
| ShuffleNetV2  | [实现](./python_developer_tools/cv/classes/ShuffleNetV2.py)|
| SqueezeNet  | [实现](./python_developer_tools/cv/classes/SqueezeNet.py)|
| Xception  | [实现](./python_developer_tools/cv/classes/Xception.py)|


# files
## [common](./python_developer_tools/files/common.py)
<table>
    <thead>
        <tr><th>名称</th><th>功能</th></tr>
    </thead>
    <tbody>
        <tr><td>get_filename_suf_pix</td><td>获取路径的文件名,后缀,父路径</td></tr>
    </tbody>
</table>

## [pickle](./python_developer_tools/files/pickle_utils.py)
<table>
    <thead>
        <tr><th>名称</th><th>功能</th></tr>
    </thead>
    <tbody>
        <tr><td>write_pkl</td><td>将数据存储为pkl</td></tr>
        <tr><td>read_pkl</td><td>读取pkl文件的内容</td></tr>
    </tbody>
</table>
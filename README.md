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
### 1. Depthwise Separable Convolution Usage
#### 1.1. Paper
["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/abs/1704.04861)

#### 1.2. Overview
![](https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/img/DepthwiseSeparableConv.png)

#### 1.3. Code
[代码来源External-Attention-pytorch](https://github.com/xmu-xiaoma666/External-Attention-pytorch#1-Depthwise-Separable-Convolution-Usage)
```python
from python_developer_tools.cv.bases.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
input=torch.randn(1,3,224,224)
dsconv=DepthwiseSeparableConvolution(3,64)
out=dsconv(input)
```
***
    
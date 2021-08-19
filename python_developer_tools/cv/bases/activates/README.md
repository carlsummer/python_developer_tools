# relu
##### code
```python
import torch.nn as nn
relu = nn.ReLU(inplace=True)
```

# LeakyReLU
##### code
```python
import torch.nn as nn
leakyrelu = nn.LeakyReLU(inplace=True)
```

# relu6
##### code
```python
import torch.nn as nn
relu6 = nn.ReLU6(inplace=True)
```

# SiLU
##### code
```python
import torch.nn as nn
silu = nn.SiLU(inplace=True)
```

# Swish
##### code
```python
import torch
import torch.nn as nn
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
swish = Swish()
```


# ACON
1. https://mp.weixin.qq.com/s/BwnsqwECR9xmUjFsxDf1AQ
2. 论文地址：https://arxiv.org/abs/2009.04759
3. 代码：https://github.com/nmaac/acon

***
# DynamicReLU
## Paper
[Dynamic ReLU](https://arxiv.org/pdf/2003.10027.pdf)

## Overview
![DynamicReLU](DynamicReLU.png)

##### Code
1. 感谢[代码来源Islanna/DynamicReLU](https://github.com/Islanna/DynamicReLU) 
2. [本库代码](DynamicReLU.py)
3. [使用demo代码](../../../../test/DynamicReLUdemo.py)
4. https://zhuanlan.zhihu.com/p/142650829
***

***
# APReLU (Adaptively-Parametric-ReLU)
## Paper
[Adaptively-Parametric-ReLU](https://ieeexplore.ieee.org/document/8998530)

## Overview
![Adaptively-Parametric-ReLU](Basic-idea-of-APReLU.png)

##### Code
1. 感谢[代码来源PlumedSerpent/Adaptively-Parametric-RELU-pytorch](https://github.com/PlumedSerpent/Adaptively-Parametric-RELU-pytorch/blob/master/APReLU.py) 
2. [本库代码](APReLU.py)
3. [使用demo代码](../../../../test/APReLUdemo.py)
4. https://zhuanlan.zhihu.com/p/274898817
5. https://github.com/zhao62/Adaptively-Parametric-ReLU
***

softmax
https://github.com/slwang9353/Period-alternatives-of-Softmax
https://arxiv.org/pdf/2108.07153.pdf
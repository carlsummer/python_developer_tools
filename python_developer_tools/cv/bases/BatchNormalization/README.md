## BN-NAS: Neural Architecture Search with Batch Normalization
1. [github代码](https://github.com/bychen515/BNNAS)
2. [论文](https://arxiv.org/abs/2108.07375)

# inplace_abn
> pip install inplace-abn <br/>
> [代码来源](https://github.com/mapillary/inplace_abn)
```python
from inplace_abn import InPlaceABN
activation="leaky_relu"
activation_param=1e-2
nf 为卷积的输出通道数
InPlaceABN(num_features=nf, activation=activation, activation_param=activation_param)
```

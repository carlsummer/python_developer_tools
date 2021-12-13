## SSM
##### Paper
[Exploiting Featureswith Split-and-Share Module](https://arxiv.org/abs/2108.04500)
##### Overview
![](../../../../temimg/SSM.png)
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

## SwishLinear
##### Paper
[MicroNet](https://arxiv.org/abs/2108.05894)
##### code
```python
from python_developer_tools.cv.bases.FC.SwishLinear import SwishLinear
import torch
x = torch.randn(2, 2048, 1, 1)
x = x.view(x.size(0), -1)
model = SwishLinear()
out = model(x)
print(out.shape)
```
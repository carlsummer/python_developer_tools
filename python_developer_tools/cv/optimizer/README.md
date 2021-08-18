# SGD
## code
```python
import torch
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

# Adam
## code
```python
import torch
optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
```

# adamw
## code
```python
import torch
optimizer = torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
```

# adadelta
## code
```python
import torch
optimizer = torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
```

# rmsprop
## code
```python
import torch
optimizer = torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
```

# [radam](RAdam.py)
## code
```python
import torch
from .RAdam import RAdam
optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, betas=(0.90, 0.999), eps=1e-08, weight_decay=1e-4)
```

# [ranger](Ranger.py)
## code
```python
import torch
from .Ranger import Ranger
optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4)
```
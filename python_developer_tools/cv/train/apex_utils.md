```shell script
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

```python
from apex import amp
import apex
...

model = resnet18()
optimizer = Adam(....)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
...
logits = model(inputs)
train_loss = criterion(logits, truth)
with amp.scale_loss(train_loss, optimizer) as scaled_loss:
    scaled_loss.backward()
optimizer.step()
```
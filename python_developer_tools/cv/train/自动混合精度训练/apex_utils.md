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

> opt_level
> O0：纯FP32训练，可以作为accuracy的baseline；
> O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
> O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算。
> O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；
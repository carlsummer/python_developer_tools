## torch_utils
[TOC]
### 固定随机种子
- 固定pytorch训练时所有的随机种子
使用方法:init_seeds(seed=0)
- torch_utils.py
```py
def init_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

    # 设置随机种子  rank = -1
    # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，
    # 导致结果不确定。如果设置初始化，则每次初始化都是固定的
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
```
### cuda变量转cpu
- 将pytorch的变量从cuda内存中移动到cpu的内存中
使用例子：
cuda2cpu(pred)
- torch_utils.py
```py
def cuda2cpu(pred):
    # 将cuda的torch变量转为cpu
    if pred.is_cuda:
        pred_cpu = pred.cpu().numpy()
    else:
        pred_cpu = pred.numpy()
    return pred_cpu
```
### 选择训练设备
- 使用例子：
select_device("0")
- torch_utils.py
```py
def select_device(device=''):
    """选择训练设备"""
    return torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')

```

### 查看cuda cudnn torch cpu 等版本号
- 查看cuda cudnn torch cpu 等版本号
- torch_utils.py
```py
def collect_env_info():
    """查看cuda cudnn torch 等版本是多少"""
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    torch_version = torch.__version__

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    has_rocm = False
    if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
        has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))  # check-template.yml depends on it
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))
```

### 根据图片样本数量计算weights
- 
- torch_utils.py
```py
def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount([labels[i]], minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights
```

### 加速训练还是追求模型性能
- reproducibility_training_speed为True表示训练速度加快，复现性;为false不能复现，提升网络性能
- torch_utils.py
```py
def init_cudnn(reproducibility_training_speed=True):
    if reproducibility_training_speed:
        # 因此方便复现、提升训练速度就：
        torch.backends.cudnn.benchmark = False
        # 虽然通过torch.backends.cudnn.benchmark = False限制了算法的选择这种不确定性，但是由于，
        # 算法本身也具有不确定性，因此可以通过设置：
        torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    else:
        # 不需要复现结果、想尽可能提升网络性能：
        torch.backends.cudnn.benchmark = True
```
### 返回全局的整个的进程数
- 返回全局的整个的进程数
- torch_utils.py
```py
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()
```
### 执行该脚本的进程的rank
- 执行该脚本的进程的rank
- torch_utils.py
```py
def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
```

### 沿给定dim维度返回输入张量input中 k 个最大值。
```python
torch.topk(input, k, dim=None, largest=True, sorted=True, out=None) -> (Tensor, LongTensor)
沿给定dim维度返回输入张量input中 k 个最大值。
如果不指定dim，则默认为input的最后一维。
如果为largest为 False ，则返回最小的 k 个值。
返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标。
如果设定布尔值sorted 为_True_，将会确保返回的 k 个值被排序。
参数:
    input (Tensor) – 输入张量
    k (int) – “top-k”中的k
    dim (int, optional) – 排序的维
    largest (bool, optional) – 布尔值，控制返回最大或最小值
    sorted (bool, optional) – 布尔值，控制返回值是否排序
    out (tuple, optional) – 可选输出张量 (Tensor, LongTensor) output buffer
```

### torch.meshgrid
```python
torch.meshgrid（）的功能是生成网格，可以用于生成坐标。函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。
其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素各列元素相同。

import torch
a = torch.tensor([1, 2, 3, 4])
print(a)
b = torch.tensor([4, 5, 6])
print(b)
x, y = torch.meshgrid(a, b)
print(x)
print(y)
 
结果显示：
tensor([1, 2, 3, 4])
tensor([4, 5, 6])
tensor([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]])
tensor([[4, 5, 6],
        [4, 5, 6],
        [4, 5, 6],
```

### 线性间距向量
```python
torch.linspace(start, end, steps=100, out=None) → Tensor
返回一个1维张量，包含在区间start和end上均匀间隔的step个点。
输出张量的长度由steps决定。
参数：
start (float) - 区间的起始点
end (float) - 区间的终点
steps (int) - 在start和end间生成的样本数
out (Tensor, optional) - 结果张量
```
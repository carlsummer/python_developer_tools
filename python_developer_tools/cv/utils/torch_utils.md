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
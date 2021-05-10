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


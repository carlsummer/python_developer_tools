

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

### 替换模型中的某一层
```python
def convert_relu_to_softplus(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus())
        else:
            convert_relu_to_softplus(child)
```
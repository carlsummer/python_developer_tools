### numpy.rollaxis
```python
numpy.rollaxis 函数向后滚动特定的轴到一个特定位置，格式如下：
numpy.rollaxis(arr, axis, start)
参数说明：
    arr：数组
    axis：要向后滚动的轴，其它轴的相对位置不会改变
    start：默认为零，表示完整的滚动。会滚动到特定位置。
Examples
--------
>> > a = np.ones((3, 4, 5, 6))
>> > np.rollaxis(a, 3, 1).shape
(3, 6, 4, 5)
>> > np.rollaxis(a, 2).shape
(5, 3, 4, 6)
>> > np.rollaxis(a, 1, 4).shape
(3, 5, 6, 4)
```

### 将numpy数据进行保存和读取
```python
np.savez_compressed(
    f"{prefix}_label.npz",
    aspect_ratio=image.shape[1] / image.shape[0],
    jmap=jmap,  # [J, H, W]    Junction heat map
    joff=joff,  # [J, 2, H, W] Junction offset within each pixel
    lmap=lmap,  # [H, W]       Line heat map with anti-aliasing
    junc=junc,  # [Na, 3]      Junction coordinate
    Lpos=Lpos,  # [M, 2]       Positive lines represented with junction indices
    Lneg=Lneg,  # [M, 2]       Negative lines represented with junction indices
    lpos=lpos,  # [Np, 2, 3]   Positive lines represented with junction coordinates
    lneg=lneg,  # [Nn, 2, 3]   Negative lines represented with junction coordinates
    lcls = np.array(classes) # [Np]
)

with np.load(self.filelist[idx]) as npz:
    lpos = npz["lpos"]
```

### 定义全0的numpy arrays
```python
np.zeros((1,128,16), dtype=np.float32)
```

### 切取arrays中的部分
```python
np.clip(
	a, 
	a_min, 
	a_max, 
	out=None):
a：输入矩阵；
a_min：被限定的最小值，所有比a_min小的数都会强制变为a_min；
a_max：被限定的最大值，所有比a_max大的数都会强制变为a_max；
out：可以指定输出矩阵的对象，shape与a相同

import numpy as np
# 一维矩阵
x= np.arange(12)
print(np.clip(x,3,8))
# 多维矩阵
y= np.arange(12).reshape(3,4)
print(np.clip(y,3,8))

[3 3 3 3 4 5 6 7 8 8 8 8]
[[3 3 3 3]
 [4 5 6 7]
 [8 8 8 8]]

lines[:, :, 0] = np.clip(lines[:, :, 0] * fx, 0, heatmap_scale[0] - 1e-4) # 人工智能分析的时候加1e-4
```

### 将某一维度的位置倒换
```python
[[[1,2],[3,4]]]->[[[2,1],[4,3]]]
lines[:, :, ::-1]
```

### numpy中np.max和np.maximum
```python
np.max(a, axis=None, out=None, keepdims=False)
　　求序列的最值
       最少接受一个参数
      axis默认为axis=0即列向,如果axis=1即横向
ex:
>> np.max([-2, -1, 0, 1, 2])
2

np.maximum(X, Y, out=None)
     X和Y逐位进行比较,选择最大值.
     最少接受两个参数
ex:
>> np.maximum([-3, -2, 0, 1, 2], 0)
array([0, 0, 0, 1, 2])
```

### 随机排列序列。
```python
np.random.permutation()
```

### 求范数
```python
np.linalg.norm(求范数)
```

### 求范数
```python
A[[i, j], :] = A[[j, i], :] # 实现了第i行与第j行的互换
```
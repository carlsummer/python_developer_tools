# CenterLoss
## 说明
1. [本库代码](CenterLoss.py)
2. [使用demo](../../../test/cv/loss/train_centerloss.py)
3. [代码来源](https://github.com/jxgu1016/MNIST_center_loss_pytorch)

### CrossEntropyLoss
> CrossEntropyLoss就是把以上Softmax–Log–NLLLoss合并成一步

# Diceloss

# focalloss

### LabelSmoothingCrossEntropy
> 有利于缓解过拟合

# OHEMloss
1. [paper](https://arxiv.org/abs/1604.03540)
2. [本库代码](ohem_loss.py)
3. [使用demo](../../../test/cv/loss/train_ohemloss.py)
4. [使用demo2](../../../test/cv/loss/train_ohemloss2.py)

# OIMloss

#topk_crossEntrophy
1. [使用demo](../../../test/cv/loss/train_topk_crossEntrophy.py)
2. [本库代码](topk_crossEntrophy.py)

# TripletLoss
> ![](classes/TripletLoss.png)<br/>
> 如上图所示，Triplet Loss 是有一个三元组<a, p, n>构成，其中<br/>
a: anchor 表示训练样本。<br/>
p: positive 表示预测为正样本。<br/>
n: negative 表示预测为负样本。<br/>
    triplet loss的作用：用于减少positive（正样本）与anchor之间的距离，扩大negative（负样本）与anchor之间的距离。基于上述三元组，<br/>
>可以构建一个positive pair <a, p>和一个negative pair <a, n>。triplet loss的目的是在一定距离（margin）上把positive pair和negative pair分开。<br/>
  所以我们希望：D(a, p) < D(a, n)。进一步希望在一定距离上（margin） 满足这个情况：D(a, p)  + margin  <  D(a, n)<br/>
## 说明
1. [说明](https://blog.csdn.net/weixin_40671425/article/details/98068190)
2. [代码](classes/TripletLoss.py)
3. [代码来源](https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py)
4. [使用demo](../../../test/cv/loss/train_tripletloss.py)

### 自己手写常用loss
https://blog.csdn.net/weixin_45209433/article/details/105141457

### NLLLoss
> https://blog.csdn.net/weixin_43593330/article/details/108622747


Lovasz-SoftMax Loss

### AutoLoss-Zero 通用的损失函数搜索框架
> 4快V100 2天时间可以搜索出最适合的损失函数

### GFocalV2 G focal loss
1. [github代码](https://github.com/implus/GFocalV2)
2. [论文](https://arxiv.org/abs/2011.12885)

### Rank & Sort Loss for Object Detection and Instance Segmentation
1. [github代码](https://github.com/kemaloksuz/RankSortLoss)
2. [论文](https://arxiv.org/abs/2107.11669)
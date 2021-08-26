### 训练技巧
1. 先用少量样本去训练最优的算法，参数，数据增强等。。然后再增加数据集
2. 对应产线的数据应该用所有的数据集进行训练，然后用模型预测和发现预测不对的图片做为验证集，取验证集表现最好的发布
3. 对每一次采用的测量进行写execl记录

### 读论文源代码的方法
1. 先将代码跑通
2. 将utils等工具类方法理解并且，提取出来
3. 理解datasets，如何进行数据处理，也就是数据预处理
4. 理解网络模型 [结合论文一起]
5. 理解loss
6. 理解网络结果出来的后处理
7. 最重要的是[耐心分析解读]。

### 推理加速
1. 优化代码细节
2. numpy操作转torch的tensor在GPU上
3. 多进程
4. onnx，onnxruntime，
5. tensorrt

### 标注
1. 人工标注一些图片
2. 训练初版模型进行预测
3. 人工清洗模型预测的图片->2->3...

### 业务数据集目录
├─任务名称
│  ├─label_datasets 最开始人工标注的数据集
│  │  └─20210801
│  │  └─20210802
│  │  └─.....
│  ├─lab_datasets 用来实验用的数据集train，test，val比例
|  |    └─训练完用模型找识别不对的图片，进行分析
│  ├─deploy_question_datasets 每次新的有问题的图片
│  ├─model 用来存放每次实验的模型


### 其他可能有用的tricks集合
> https://liumin.blog.csdn.net/
> https://github.com/shanglianlm0525/CvPytorch
> https://github.com/shanglianlm0525/PyTorch-Networks
> https://github.com/bobo0810/PytorchNetHub
> [tricks 收集](https://github.com/xmu-xiaoma666/External-Attention-pytorch#23-Residual-Attention-Usage)
> [代码解读](https://blog.csdn.net/shenjianhua005/article/details/117414292)
> https://github.com/YehLi/xmodaler--https://xmodaler.readthedocs.io/en/latest/
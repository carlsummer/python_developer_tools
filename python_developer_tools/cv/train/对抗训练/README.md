# 对抗训练
> [code](https://github.com/Harry24k/adversarial-attacks-pytorch)
![](adversarialattackspytorchmaster/pic/adv_kor.png)
> 对抗训练就是生成对抗样本然后在用对抗样本进行训练<br/>
> 训练完后有两个评判标准一个是之前没加对抗的标准准确率Standard Accuracy<br/>
> 一个是加了后的对抗准确率Robust Accuracy<br/>
> [使用demo](adversarialattackspytorchmaster/demos/White Box Attack (ImageNet).ipynb)

# FGSM
1. [code](FGSM.py)

# [所有能用并且验证过的对抗训练使用demo代码](../../../../test/cv/train/对抗训练/test_GN_shufflenet.py)
# [所有能用并且验证过的对抗训练使用demo代码](../../../../test/cv/train/对抗训练/test_GN_resnet18.py)
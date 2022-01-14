# 对抗训练
> [code](https://github.com/Harry24k/adversarial-attacks-pytorch)
![](adversarialattackspytorchmaster/pic/adv_kor.png)
> 对抗训练就是生成对抗样本然后在用对抗样本进行训练<br/>
> 训练完后有两个评判标准一个是之前没加对抗的标准准确率Standard Accuracy<br/>
> 一个是加了后的对抗准确率Robust Accuracy<br/>
> [使用demo](adversarialattackspytorchmaster/demos/White Box Attack (ImageNet).ipynb)

# FGSM
1. [code](FGSM.py)
2. [所有能用并且验证过的对抗训练使用demo代码](../../../../test/cv/train/对抗训练/test_GN_shufflenet.py)
3. [所有能用并且验证过的对抗训练使用demo代码](../../../../test/cv/train/对抗训练/test_GN_resnet18.py)

# advertorch
[github地址](https://github.com/BorealisAI/advertorch)
[代码示例](advertorch_examples)
```shell
pip install advertorch
```
1. GradientAttack
2. GradientSignAttack
3. FGM
4. FGSM
5. FastFeatureAttack
6. L2BasicIterativeAttack
7. LinfBasicIterativeAttack
8. PGDAttack
9. LinfPGDAttack
10. L2PGDAttack
11. L1PGDAttack
12. SparseL1DescentAttack
13. MomentumIterativeAttack
14. L2MomentumIterativeAttack
15. LinfMomentumIterativeAttack
16. CarliniWagnerL2Attack
17. ElasticNetL1Attack
18. DDNL2Attack
19. DeepfoolLinfAttack
20. LBFGSAttack
21. SinglePixelAttack
22. LocalSearchAttack
23. SpatialTransformAttack
24. JacobianSaliencyMapAttack
25. JSMA
26. LinfSPSAAttack
27. FABAttack
28. LinfFABAttack
39. L2FABAttack
30. L1FABAttack
31. ChooseBestAttack

# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 9:56 AM
# @File:scheduler

# Scheduler https://arxiv.org/pdf/1812.01187.pdf
# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
linear_lr = True
if linear_lr:
    lf = lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
else:
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - hyp['lrf']) + hyp[
        'lrf']  # cosine 1->hyp['lrf']
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# scheduler = lr_scheduler.MultiStepLR(optimizer, [30,60,90,120,150], gamma=0.1, last_epoch=-1)

# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=False,threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-7,eps=1e-8)

# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

# scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

# scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=hyp['lr0'], max_lr=0.1,mode='exp_range', gamma=0.98,step_size_up=10, cycle_momentum=False)

loss.backward()
optimizer.step()
scheduler.step()  # 调整学习率
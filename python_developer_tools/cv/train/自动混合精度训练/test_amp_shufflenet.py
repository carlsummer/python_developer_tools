# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/26/2021 8:56 AM
# @File:test_FGSM
import os
import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from python_developer_tools.cv.classes.transferTorch import shufflenet_v2_x0_5
from python_developer_tools.cv.utils.torch_utils import init_seeds

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    #使用前 占用显存大小1521MiB 训练时间：122.50208616256714  精度41.169998 %
    #使用后 占用显存大小1385MiB 训练时间：154.03511953353882  精度41.180000 %
    starttime = time.time()
    
    root_dir = "/home/zengxh/datasets"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    epochs = 50
    batch_size = 1024
    num_workers = 8
    classes = 10

    init_seeds(1024)

    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = shufflenet_v2_x0_5(classes, True)
    model.cuda()
    model.train()

    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # 在训练最开始之前实例化一个GradScaler对象
    scaler = GradScaler()

    for epoch in range(epochs):
        train_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # 前向过程(model + loss)开启 autocast
            with autocast():
                # forward
                outputs = model(inputs)
                # loss
                loss = criterion(outputs, labels)
            # backward 反向传播在autocast上下文之外
            # Scales loss. 为了梯度放大.
            scaler.scale(loss).backward()
            # update weights
            # scaler.step() 首先把梯度的值unscale回来.
            # 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
            # 否则，忽略step调用，从而保证权重不更新（不被破坏）
            scaler.step(optimizer)
            # 准备着，看是否要增大scaler
            scaler.update()
            
            # # forward
            # outputs = model(inputs)
            # # loss
            # loss = criterion(outputs, labels)
            # # backward
            # loss.backward()
            # # update weights
            # optimizer.step()

            # print statistics
            train_loss += loss

        scheduler.step()
        print('%d/%d loss: %.6f' % (epochs, epoch + 1, train_loss / len(trainset)))
        
    print("训练所花费时间:", time.time() - starttime)  # 146.15674996376038
    
    correct = 0
    model.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the network on the 10000 test images: %.6f %%' % (100 * correct / len(testset)))

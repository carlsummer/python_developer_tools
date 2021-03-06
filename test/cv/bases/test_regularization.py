# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/23/2021 11:14 AM
# @File:test_regularization
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from python_developer_tools.cv.bases.regularization import Regularization
from python_developer_tools.cv.utils.torch_utils import init_seeds

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def shufflenet_v2_x0_5(nc, pretrained):
    model_ft = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, nc)
    return model_ft


if __name__ == '__main__':
    root_dir = "/home/zengxh/datasets" # 定义好会自己在这个目录创建数据集
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    epochs = 50
    batch_size = 1024
    num_workers = 8
    classes = 10
    lr = 0.001
    weight_decay = 0.1 # 调用正则化的大小

    init_seeds(1024)

    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = shufflenet_v2_x0_5(classes, True)
    model.cuda()
    model.train()

    reg_loss = Regularization(model, weight_decay, p=2).cuda()
    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            # loss
            loss = criterion(outputs, labels)
            loss = loss + reg_loss(model)
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # print statistics
            train_loss += loss

        scheduler.step()
        print('%d/%d loss: %.3f' % (epochs, epoch + 1, train_loss / len(trainset)))

    correct = 0
    model.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / len(testset)))
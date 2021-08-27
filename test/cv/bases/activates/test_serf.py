# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/14/2021 3:19 PM
# @File:demo
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

from python_developer_tools.cv.bases.activates.DynamicReLU import DyReLUA, DyReLUB, DyReLUC, convert_relu_to_DyReLU
from python_developer_tools.cv.bases.activates.Serf import Serf, convert_relu_to_serf
from python_developer_tools.cv.utils.torch_utils import init_seeds

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class shufflenet_v2_x0_5M(nn.Module):
    def __init__(self,nc,pretrained=True):
        super(shufflenet_v2_x0_5M, self).__init__()
        self.model_ft = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained)
        # 将relu替换为DyReLUA
        self.model_ft = convert_relu_to_serf(self.model_ft)

        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc = nn.Linear(num_ftrs, nc)

    def forward(self,x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.maxpool(x)
        x = self.model_ft.stage2(x)
        x = self.model_ft.stage3(x)
        x = self.model_ft.stage4(x)
        x = self.model_ft.conv5(x)
        x = x.mean([2, 3])  # globalpool
        out = self.model_ft.fc(x)
        return out



if __name__ == '__main__':
    """
    ReLU 41%
    DyReLUA 42 %
    DyReLUB 41 %
    DyReLUC 40 %
    """
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

    model = shufflenet_v2_x0_5M(classes, True)
    model.cuda()
    model.train()

    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
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
            # backward
            loss.backward()
            # update weights
            optimizer.step()

            # print statistics
            train_loss += loss

        scheduler.step()
        print('%d/%d loss: %.6f' % (epochs, epoch + 1, train_loss / len(trainset)))

    correct = 0
    model.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / len(testset)))
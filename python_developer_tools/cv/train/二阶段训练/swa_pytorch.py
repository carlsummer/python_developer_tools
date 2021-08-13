# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/13/2021 11:08 AM
# @File:swa_pytorch
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR

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
    # 使用之前：41%
    # 使用swa之后：69%
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    epochs = 50
    swa_start_epoch = 25
    batch_size = 1024
    num_workers = 8
    classes = 10

    init_seeds(1024)

    trainset = torchvision.datasets.CIFAR10(root=os.getcwd(), train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=os.getcwd(), train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = shufflenet_v2_x0_5(classes, True)
    model.cuda()
    model.train()

    swa_model = torch.optim.swa_utils.AveragedModel(model)

    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    swa_scheduler = SWALR(optimizer, swa_lr=0.05)

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
        if epoch > swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        print('%d/%d loss: %.3f' % (epochs, epoch + 1, train_loss / len(trainset)))

    torch.optim.swa_utils.update_bn(trainloader, swa_model,device=inputs.device)

    correct_model = 0
    correct_swa_model = 0
    model.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        images=images.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_model += (predicted.cpu() == labels).sum()

        preds = swa_model(images)
        _, predicted = torch.max(outputs.data, 1)
        correct_swa_model += (predicted.cpu() == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_model / len(testset)))
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct_swa_model / len(testset)))




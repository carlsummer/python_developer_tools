# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/13/2021 11:20 AM
# @File:train_cifar10
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

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
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    epochs = 5
    batch_size = 256
    num_workers = 8
    classes = 10

    trainset = torchvision.datasets.CIFAR10(root=os.getcwd(), train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=os.getcwd(), train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = shufflenet_v2_x0_5(classes, True)
    model.cuda()
    model.train()

    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
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
            running_loss += loss
        print('%d/%d loss: %.3f' % (epochs, epoch + 1, running_loss / len(trainset)))

    correct = 0
    model.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / len(testset)))

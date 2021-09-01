# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/26/2021 8:56 AM
# @File:test_FGSM
import copy

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from python_developer_tools.cv.classes.transferTorch import resnet152, resnet18, shufflenet_v2_x0_5
from python_developer_tools.cv.train.对抗训练.FGSM import fgsm_attack
from python_developer_tools.cv.utils.torch_utils import init_seeds
from python_developer_tools.cv.train.对抗训练.adversarialattackspytorchmaster.torchattacks import *

transform = transforms.Compose(
    [transforms.ToTensor(),# ToTensor : [0, 255] -> [0, 1]
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if __name__ == '__main__':
    # GN 42.759998 %  FGSM 43.939999 %  FFGSM 44.450001 %
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

    model = shufflenet_v2_x0_5(classes, True).cuda().train()

    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        train_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs = inputs.cuda()  # 72.070000 %
            labels = labels.cuda()
            model.train()
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
            # train_loss += loss

            model.eval()
            atk = FFGSM(model, eps=8 / 255, alpha=10 / 255)
            inputs2 = atk(inputs, labels)  # 72.190002 %
            model.train()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs2)
            # loss
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            # update weights
            optimizer.step()
            # print statistics
            train_loss += loss

        scheduler.step()

        print('%d/%d loss: %.6f' % (epochs, epoch + 1, train_loss / (len(trainset) * 2)))

    # Standard Accuracy
    correct = 0
    model.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the network on the 10000 test images: %.6f %%' % (100 * correct / len(testset)))

    # Robust Accuracy
    correct = 0
    model.eval()
    atk.set_training_mode(training=False)
    atk.save(data_loader=testloader, save_path="testloader.pt", verbose=True)
    adv_images, adv_labels = torch.load("testloader.pt")
    adv_data = TensorDataset(adv_images.float() / 255, adv_labels)
    adv_testloader = DataLoader(adv_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for j, (images, labels) in tqdm(enumerate(adv_testloader)):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Robust Accuracy of the network on the 10000 test images: %.6f %%' % (100 * correct / len(testset)))
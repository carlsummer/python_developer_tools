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
    # GN 40.820000 %
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

    model = resnet18(classes, True).cuda().train()

    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    """
    找最优的对抗方式
    atks = [
        FGSM(model, eps=8 / 255),
        BIM(model, eps=8 / 255, alpha=2 / 255, steps=100),
        RFGSM(model, eps=8 / 255, alpha=2 / 255, steps=100),
        CW(model, c=1, lr=0.01, steps=100, kappa=0),
        PGD(model, eps=8 / 255, alpha=2 / 225, steps=100, random_start=True),
        PGDL2(model, eps=1, alpha=0.2, steps=100),
        EOTPGD(model, eps=8 / 255, alpha=2 / 255, steps=100, eot_iter=2),
        FFGSM(model, eps=8 / 255, alpha=10 / 255),
        TPGD(model, eps=8 / 255, alpha=2 / 255, steps=100),
        MIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=100, decay=0.1),
        VANILA(model),
        GN(model, sigma=0.1),
        APGD(model, eps=8 / 255, steps=100, eot_iter=1, n_restarts=1, loss='ce'),
        APGD(model, eps=8 / 255, steps=100, eot_iter=1, n_restarts=1, loss='dlr'),
        APGDT(model, eps=8 / 255, steps=100, eot_iter=1, n_restarts=1),
        FAB(model, eps=8 / 255, steps=100, n_classes=10, n_restarts=1, targeted=False),
        FAB(model, eps=8 / 255, steps=100, n_classes=10, n_restarts=1, targeted=True),
        Square(model, eps=8 / 255, n_queries=5000, n_restarts=1, loss='ce'),
        AutoAttack(model, eps=8 / 255, n_classes=10, version='standard'),
        OnePixel(model, pixels=5, inf_batch=50),
        DeepFool(model, steps=100),
        DIFGSM(model, eps=8 / 255, alpha=2 / 255, steps=100, diversity_prob=0.5, resize_rate=0.9)
    ]
    bestatk = None
    bestRobustAcc = 0
    for atk in atks:
        print("-" * 70)
        print(atk)
        correct = 0
        model.eval()
        for j, (images, labels) in tqdm(enumerate(trainloader)):
            adv_images = atk(images, labels)
            outputs = model(adv_images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted.cpu() == labels).sum()
        bestRobustAcc_now = correct / len(trainset)
        print('Robust Accuracy: %.4f %%' % (bestRobustAcc_now))
        if bestRobustAcc < bestRobustAcc_now:
            bestatk = atk
            bestRobustAcc = bestRobustAcc_now
    """

    # 使用带个对抗方式
    model.eval()
    atk = GN(model, sigma=0.1)
    atk.set_return_type('int')  # Save as integer.
    # atk.save(data_loader=trainloader, save_path="trainloader.pt", verbose=True)
    atk.save(data_loader=testloader, save_path="testloader.pt", verbose=True)
    # adv_images, adv_labels = torch.load("trainloader.pt")
    # adv_data = TensorDataset(adv_images.float() / 255, adv_labels)
    # adv_trainloader = DataLoader(adv_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True)
    adv_images, adv_labels = torch.load("testloader.pt")
    adv_data = TensorDataset(adv_images.float() / 255, adv_labels)
    adv_testloader = DataLoader(adv_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    for epoch in range(epochs):
        train_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            model.train()
            inputs = inputs.cuda()  # 72.070000 %
            labels = labels.cuda()

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

            model.eval()
            inputs = atk(inputs, labels) #72.190002 %

        scheduler.step()
        print('%d/%d loss: %.6f' % (epochs, epoch + 1, train_loss / len(trainset)))

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
    for j, (images, labels) in tqdm(enumerate(adv_testloader)):
        outputs = model(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Robust Accuracy of the network on the 10000 test images: %.6f %%' % (100 * correct / len(testset)))
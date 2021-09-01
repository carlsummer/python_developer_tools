# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/13/2021 11:20 AM
# @File:train_cifar10
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from python_developer_tools.cv.classes.transferTorch import resnet18
from python_developer_tools.cv.utils.torch_utils import init_seeds
from python_developer_tools.cv.loss.ohem_loss import OhemLargeMarginLoss,OhemCELoss
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if __name__ == '__main__':
    # model1 73.919998 % model2  72.720001 %
    root_dir = "/home/zengxh/datasets"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    epochs = 50
    batch_size = 1024
    num_workers = 8
    classes = 10
    img_width,img_height = 32,32

    init_seeds(1024)

    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model1 = resnet18(classes, True)
    model1.cuda()
    model1.train()

    model2 = resnet18(classes, True)
    model2.cuda()
    model2.train()

    criteria1 = OhemLargeMarginLoss(score_thresh=0.7, n_min=batch_size*img_width*img_height//batch_size).cuda()
    criteria2 = OhemCELoss(score_thresh=0.7, n_min=batch_size*img_width*img_height//batch_size).cuda()
    # SGD with momentum
    optimizer1 = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.001, momentum=0.9)
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=epochs)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(epochs):
        train_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            # forward
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            # loss
            loss1 = criteria1(outputs1, labels)
            loss2 = criteria2(outputs2, labels)
            loss = loss1 + loss2
            # backward
            loss.backward()
            # update weights
            optimizer1.step()
            optimizer2.step()

            # print statistics
            train_loss += loss

        scheduler1.step()
        scheduler2.step()
        print('%d/%d loss: %.6f' % (epochs, epoch + 1, train_loss / len(trainset)))

    correct = 0
    model1.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        outputs = model1(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the network on the 10000 test images: %.6f %%' % (100 * correct / len(testset)))

    correct = 0
    model2.eval()
    for j, (images, labels) in tqdm(enumerate(testloader)):
        outputs = model2(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted.cpu() == labels).sum()
    print('Accuracy of the network on the 10000 test images: %.6f %%' % (100 * correct / len(testset)))

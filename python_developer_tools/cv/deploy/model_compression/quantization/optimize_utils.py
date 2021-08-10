# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/21 9:54
# @Author : liumin
# @File : optimize_utils.py

import os
import time

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

"""
融合模块：将操作/模块融合进一个模块来获得高性能与速度。通过torch.quantization.fuse_modules() API，
可以将成列表的子模块进行融合，目前仅支持[Conv, ReLU],[Conv, BatchNorm], [Conv, BatchNorm, ReLU], [Linear, ReLU]
"""
def fusebn(model):
    # print('Fusing layers... ')
    modules_to_fuse = [['conv1.0', 'conv1.1', 'conv1.2'],

                       ['stage2.0.branch1.0', 'stage2.0.branch1.1'],
                       ['stage2.0.branch1.2', 'stage2.0.branch1.3', 'stage2.0.branch1.4'],
                       ['stage2.0.branch2.0', 'stage2.0.branch2.1', 'stage2.0.branch2.2'],
                       ['stage2.0.branch2.3', 'stage2.0.branch2.4'],
                       ['stage2.0.branch2.5', 'stage2.0.branch2.6', 'stage2.0.branch2.7'],
                       ['stage2.1.branch2.0', 'stage2.1.branch2.1', 'stage2.1.branch2.2'],
                       ['stage2.1.branch2.3', 'stage2.1.branch2.4'],
                       ['stage2.1.branch2.5', 'stage2.1.branch2.6', 'stage2.1.branch2.7'],
                       ['stage2.2.branch2.0', 'stage2.2.branch2.1', 'stage2.2.branch2.2'],
                       ['stage2.2.branch2.3', 'stage2.2.branch2.4'],
                       ['stage2.2.branch2.5', 'stage2.2.branch2.6', 'stage2.2.branch2.7'],
                       ['stage2.3.branch2.0', 'stage2.3.branch2.1', 'stage2.3.branch2.2'],
                       ['stage2.3.branch2.3', 'stage2.3.branch2.4'],
                       ['stage2.3.branch2.5', 'stage2.3.branch2.6', 'stage2.3.branch2.7'],

                       ['stage3.0.branch1.0', 'stage3.0.branch1.1'],
                       ['stage3.0.branch1.2', 'stage3.0.branch1.3', 'stage3.0.branch1.4'],
                       ['stage3.0.branch2.0', 'stage3.0.branch2.1', 'stage3.0.branch2.2'],
                       ['stage3.0.branch2.3', 'stage3.0.branch2.4'],
                       ['stage3.0.branch2.5', 'stage3.0.branch2.6', 'stage3.0.branch2.7'],
                       ['stage3.1.branch2.0', 'stage3.1.branch2.1', 'stage3.1.branch2.2'],
                       ['stage3.1.branch2.3', 'stage3.1.branch2.4'],
                       ['stage3.1.branch2.5', 'stage3.1.branch2.6', 'stage3.1.branch2.7'],
                       ['stage3.2.branch2.0', 'stage3.2.branch2.1', 'stage3.2.branch2.2'],
                       ['stage3.2.branch2.3', 'stage3.2.branch2.4'],
                       ['stage3.2.branch2.5', 'stage3.2.branch2.6', 'stage3.2.branch2.7'],
                       ['stage3.3.branch2.0', 'stage3.3.branch2.1', 'stage3.3.branch2.2'],
                       ['stage3.3.branch2.3', 'stage3.3.branch2.4'],
                       ['stage3.3.branch2.5', 'stage3.3.branch2.6', 'stage3.3.branch2.7'],
                       ['stage3.4.branch2.0', 'stage3.4.branch2.1', 'stage3.4.branch2.2'],
                       ['stage3.4.branch2.3', 'stage3.4.branch2.4'],
                       ['stage3.4.branch2.5', 'stage3.4.branch2.6', 'stage3.4.branch2.7'],
                       ['stage3.5.branch2.0', 'stage3.5.branch2.1', 'stage3.5.branch2.2'],
                       ['stage3.5.branch2.3', 'stage3.5.branch2.4'],
                       ['stage3.5.branch2.5', 'stage3.5.branch2.6', 'stage3.5.branch2.7'],
                       ['stage3.6.branch2.0', 'stage3.6.branch2.1', 'stage3.6.branch2.2'],
                       ['stage3.6.branch2.3', 'stage3.6.branch2.4'],
                       ['stage3.6.branch2.5', 'stage3.6.branch2.6', 'stage3.6.branch2.7'],
                       ['stage3.7.branch2.0', 'stage3.7.branch2.1', 'stage3.7.branch2.2'],
                       ['stage3.7.branch2.3', 'stage3.7.branch2.4'],
                       ['stage3.7.branch2.5', 'stage3.7.branch2.6', 'stage3.7.branch2.7'],

                       ['stage4.0.branch1.0', 'stage4.0.branch1.1'],
                       ['stage4.0.branch1.2', 'stage4.0.branch1.3', 'stage4.0.branch1.4'],
                       ['stage4.0.branch2.0', 'stage4.0.branch2.1', 'stage4.0.branch2.2'],
                       ['stage4.0.branch2.3', 'stage4.0.branch2.4'],
                       ['stage4.0.branch2.5', 'stage4.0.branch2.6', 'stage4.0.branch2.7'],
                       ['stage4.1.branch2.0', 'stage4.1.branch2.1', 'stage4.1.branch2.2'],
                       ['stage4.1.branch2.3', 'stage4.1.branch2.4'],
                       ['stage4.1.branch2.5', 'stage4.1.branch2.6', 'stage4.1.branch2.7'],
                       ['stage4.2.branch2.0', 'stage4.2.branch2.1', 'stage4.2.branch2.2'],
                       ['stage4.2.branch2.3', 'stage4.2.branch2.4'],
                       ['stage4.2.branch2.5', 'stage4.2.branch2.6', 'stage4.2.branch2.7'],
                       ['stage4.3.branch2.0', 'stage4.3.branch2.1', 'stage4.3.branch2.2'],
                       ['stage4.3.branch2.3', 'stage4.3.branch2.4'],
                       ['stage4.3.branch2.5', 'stage4.3.branch2.6', 'stage4.3.branch2.7'],

                       ['conv5.0', 'conv5.1']]

    return torch.quantization.fuse_modules(model, modules_to_fuse)


def val_model(model, dataloader):
    print('-' * 10)
    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode

    running_corrects = 0
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print('Acc: {:.4f}'.format(epoch_acc))

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__ == '__main__':
    data_dir = '/home/deploy/datasets/hymenoptera'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model = models.shufflenet_v2_x0_5(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('shufflenet_v2_x0_5_train.pt', map_location='cpu'))
    model.to(device)
    model.eval()

    since = time.time()
    val_model(model, dataloaders['val'])
    time_elapsed = time.time() - since
    print(" used:", time_elapsed)
    # print(model)

    model = fusebn(model)
    since = time.time()
    val_model(model, dataloaders['val'])
    time_elapsed = time.time() - since
    print(" used:", time_elapsed)
    # print(model)


    """
    torch.quantization.quantize_dynamic只能是CPU
    torch.device('cpu')
    初始化一个RNN模型，里面包含了LSTM层和全连接层，使用torch.quantization.quantize_dynamic对模型进行量化。
    import torch.quantization
    quantized_model = torch.quantization.quantize_dynamic(
        rnn, {nn.Linear}, dtype=torch.qint8                     #rnn为模型的名字，我们只量化线性层
    )
    print(quantized_model)
    如果想量化线性层和LSTM层，将{nn.Linear}改为{nn.Linear,nn.LSTM}即可
    dtype=torch.qint8 表示量化为有符号8位数，也可以选择无符号8位数quint8
    """
    # model = torch.quantization.quantize_dynamic(
    #     model, {nn.Linear}, dtype=torch.qint8  # rnn为模型的名字，我们只量化线性层
    # )
    device = torch.device('cpu')
    model.to(device)
    model = torch.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)
    since = time.time()
    val_model(model, dataloaders['val'])
    time_elapsed = time.time() - since
    print(" used:", time_elapsed)
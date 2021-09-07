# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/13/2021 11:20 AM
# @File:train_cifar10
import os
import sys

sys.path.append('/home/zengxh/workspace/python_developer_tools')  # 绝对路径
import argparse
import time
from loguru import logger
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from python_developer_tools.cv.classes.transferTorch import shufflenet_v2_x0_5
from python_developer_tools.cv.utils.torch_utils import init_seeds, select_device

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# torch.distributed.get_world_size()：获取全局并行数
# torch.distributed.new_group()：使用 world 的子集，创建新组，用于集体通信等
# torch.distributed.get_rank()：获取当前进程的序号，用于进程间通讯。
# torch.distributed.local_rank()：获取本台机器上的进程的序号

@logger.catch
def main():
    # DDP
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='DDP parameter, do not modify')  # local_rank 不等于-1，启动命令 torch.distributed.launch会自动赋值 分布式训练
    opt = parser.parse_args()

    # 41.139999
    starttime = time.time()
    epochs = 50
    batch_size = 1024
    num_workers = 8
    classes = 10

    # DDP init start
    local_rank = opt.local_rank
    total_batch_size = batch_size
    device = "0,1"
    # WORLD_SIZE由torch.distributed.launch.py产生，具体数值为 nproc_per_node*node(主机数，这里为1), opt.world_size指进程总数，在这里就是我们使用的卡数
    # rank指进程序号，local_rank指本地序号，两者的区别在于前者用于进程间通讯，后者用于本地设备分配,
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    device = select_device(device, batch_size=batch_size)

    assert torch.cuda.device_count() > local_rank
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    # local_rank = torch.distributed.get_rank() dist.init_process_group 后才能获得
    assert batch_size % world_size == 0, '--batch-size must be multiple of CUDA device count'
    batch_size = total_batch_size // world_size
    print(world_size, global_rank, local_rank, device, batch_size)
    init_seeds(1024+local_rank)
    # DDP init end

    root_dir = "/home/zengxh/datasets"
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    trainset = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    # DDP
    # 使用DistributedSampler，DDP帮我们把细节都封装起来了。
    # DistributedSampler的实现方式是，不同进程会使用一个相同的随机数种子，这样shuffle出来的东西就能确保一致。
    # 具体实现上，DistributedSampler使用当前epoch作为随机数种子，从而使得不同epoch下有不同的shuffle结果。所以，
    # 记得每次 epoch 开始前都要调 用一下 sampler 的 set_epoch 方法，这样才能让数据集随机 shuffle 起来。
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=train_sampler,
                                              num_workers=num_workers, shuffle=False,
                                              pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = shufflenet_v2_x0_5(classes, True)

    # DDP
    # 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank
                # broadcast_buffers=False 在转发功能开始时启用模块的同步（广播）缓冲区的标志。default=True
                )

    criterion = nn.CrossEntropyLoss()
    # SGD with momentum
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()

        # DDP 设置sampler的epoch，DistributedSampler需要这个来维持各个进程之间的相同随机数种子
        trainloader.sampler.set_epoch(epoch)
        # 后面这部分，则与原来完全一致了。
        train_loss = 0.0
        for i, (inputs, labels) in tqdm(enumerate(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)

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

    print("训练所花费时间:", time.time() - starttime) # 285.5857148170471

    # DDP
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 我只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:  #local_rank
        # 保存模型
        saved_modelpath = "saved_model.pth"
        torch.save(model.module.state_dict(), saved_modelpath)
        # 加载保存好的模型
        model = shufflenet_v2_x0_5(classes, True)
        checkpoint = torch.load(saved_modelpath, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()
        # 测试结果
        correct = 0
        for j, (images, labels) in tqdm(enumerate(testloader)):
            # 指定设备0
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted.cpu() == labels).sum()
        logger.debug('Accuracy of the network on the 10000 test images: %.6f %%' % (100 * correct / len(testset)))
    else:
        # 不是第一块卡的model进行删除
        dist.destroy_process_group()
    torch.cuda.empty_cache()
    return 0

if __name__ == '__main__':
    #  /home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 -m torch.distributed.launch --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29501 /home/zengxh/workspace/python_developer_tools/python_developer_tools/cv/train/不同数量卡训练/单机多卡.py
    main()

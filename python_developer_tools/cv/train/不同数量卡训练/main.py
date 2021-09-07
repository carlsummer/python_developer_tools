import argparse
import os
import random

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LinModel(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        out = F.softmax(out, dim=-1)
        return out


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def build_fake_data(size=1000):
    x1 = [(random.uniform(0, 0.5), 0) for i in range(size // 2)]
    x2 = [(random.uniform(0.5, 1), 1) for i in range(size // 2)]
    return x1 + x2


def evaluate(valid_loader):
    net.eval()
    with torch.no_grad():
        cnt = 0
        total = 0
        for inputs, labels in valid_loader:
            inputs, labels = inputs.unsqueeze(1).float().cuda(), labels.long().cuda()
            output = net(inputs)
            predict = torch.argmax(output, dim=1)
            cnt += (predict == labels).sum().item()
            total += len(labels)
            # print(f'right = {(predict == labels).sum()}')
        cnt = torch.Tensor([cnt]).to(inputs.device)
        total = torch.Tensor([total]).to(inputs.device)
        reduced_param = torch.cat((cnt.view(1), total.view(1)))
        cnt = reduced_param[0].item()
        total = reduced_param[1].item()
    return cnt, total


def set_random_seed(seed):
    """Set random seed for reproducability."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help="local gpu id")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--lr', type=float, default=0.1, help="learn rate")
parser.add_argument('--epochs', type=int, default=100000, help="train epoch")
parser.add_argument('--seed', type=int, default=40, help="train epoch")


args = parser.parse_args()
args.world_size = int(os.getenv("WORLD_SIZE", '1'))

set_random_seed(args.seed)
dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(args.local_rank)
global_rank = dist.get_rank()

print(f'global_rank = {global_rank} local_rank = {args.local_rank} world_size = {args.world_size}')

net = LinModel(1, 2)
net.cuda()
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

trainset = build_fake_data(size=10000)
validset = build_fake_data(size=10000)

train_sampler = DistributedSampler(trainset)
valid_sampler = DistributedSampler(validset)

train_loader = DataLoader(trainset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          sampler=train_sampler)

valid_loader = DataLoader(validset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          pin_memory=True,
                          sampler=valid_sampler)

criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(), lr=args.lr)

net.train()
for e in range(int(args.epochs)):

    for idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.unsqueeze(1).float().cuda()
        labels = labels.long().cuda()
        output = net(inputs)
        loss = criterion(output, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        reduce_loss(loss, global_rank, args.world_size)
        # if idx % 10 == 0 and global_rank == 0:
        #     print('Epoch: {} step: {} loss: {}'.format(e, idx, loss.item()))
    cnt, total = evaluate(valid_loader)
    if global_rank == 0:
        print(f'epoch {e} || eval accuracy: {cnt / total}')

# if global_rank == 0:
#     print(f'net weight = {net.state_dict()}')

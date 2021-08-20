import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader, sampler
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

kq_value = []
iter_counter = 0
pro_root = './'

dtype = torch.float32
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_imagenet(root, train = True, transform = None, target_transform = None):
    if train:
        root = os.path.join(root, 'train')
    else:
        root = os.path.join(root, 'val')
    return datasets.ImageFolder(root = root,
                               transform = transform,
                               target_transform = target_transform)

def cosmax(input_tensor, dim=-1):
    input_tensor = torch.cos(input_tensor)
    sum_sin = input_tensor.sum(dim).unsqueeze(2).repeat(1, 1, input_tensor.shape[dim])
    out_put = input_tensor / sum_sin
    return out_put


def sirenmax(input_tensor, dim=-1):
    input_tensor = (torch.sin(input_tensor) + 1) / (2 - 2 * torch.sin(input_tensor))
    sum_sin = input_tensor.sum(dim).unsqueeze(2).repeat(1, 1, input_tensor.shape[dim])
    out_put = input_tensor / sum_sin
    return out_put


def sinmax(input_tensor, dim=-1):
    input_tensor = 1 + torch.sin(10 * input_tensor)
    sum_sin = input_tensor.sum(dim).unsqueeze(2).repeat(1, 1, input_tensor.shape[dim])
    out_put = input_tensor / sum_sin
    return out_put


def sin_2_max(input_tensor, dim=-1):
    input_tensor = torch.sin(input_tensor + 0.25 * np.pi) ** 2
    sum_sin = input_tensor.sum(dim).unsqueeze(2).repeat(1, 1, input_tensor.shape[dim])
    out_put = input_tensor / sum_sin
    return out_put


def sin_2_max_no_move(input_tensor, dim=-1):
    input_tensor = torch.sin(input_tensor) ** 2
    sum_sin = input_tensor.sum(dim).unsqueeze(2).repeat(1, 1, input_tensor.shape[dim])
    out_put = input_tensor / sum_sin
    return out_put


def map_norm(kq_maps, eps=1e-05, dim=-1):
    mean_map = kq_maps.mean(dim).unsqueeze(2).repeat(1, 1, kq_maps.shape[dim])
    var_map = kq_maps.var(dim).unsqueeze(2).repeat(1, 1, kq_maps.shape[dim])
    kq_maps = (kq_maps - mean_map) / (torch.sqrt(var_map + eps))
    return kq_maps


class TransformSingle(nn.Module):
    def __init__(self, head, d_model, use_pro='softmax', mode='search'):
        super(TransformSingle, self).__init__()
        self.d_model = d_model
        self.heads = head
        assert d_model % head == 0
        self.d_per_head = d_model // head
        self.QVK = nn.Linear(d_model, d_model * 3)
        self.relu_QVK = nn.ReLU(inplace=True)
        self.Q_to_heads = nn.Linear(d_model, d_model)
        self.K_to_heads = nn.Linear(d_model, d_model)
        self.V_to_heads = nn.Linear(d_model, d_model)
        self.concat_heads = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model)
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        self.use_pro = use_pro
        self.mode = mode

    def forward(self, x):
        res = x
        x = self.norm(x)
        QVK = self.QVK(x)
        QVK = self.relu_QVK(QVK)
        Q = QVK[:, :, 0: self.d_model]
        K = QVK[:, :, self.d_model: 2 * self.d_model]
        V = QVK[:, :, 2 * self.d_model: 3 * self.d_model]
        scores_merge = []
        heads_v = []
        for h in range(self.heads):
            q = Q[:, :, h * self.d_per_head: h * self.d_per_head + self.d_per_head]
            k = K[:, :, h * self.d_per_head: h * self.d_per_head + self.d_per_head]
            v = V[:, :, h * self.d_per_head: h * self.d_per_head + self.d_per_head]
            q_mul_k = torch.matmul(q, k.transpose(2, 1))

            if self.mode == 'run' :
                global kq_value, iter_counter
                if iter_counter % 1000 == 0:
                    save_kq = q_mul_k.cpu()
                    kq_value.append(save_kq.detach().numpy())
                iter_counter += 1

            if self.use_pro == "softmax":
                scores = F.softmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            if self.use_pro == "norm_softmax":
                q_mul_k = map_norm(q_mul_k)
                scores = F.softmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)

            if self.use_pro == "sinmax":
                scores = sinmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            if self.use_pro == "norm_sinmax":
                q_mul_k = map_norm(q_mul_k)
                scores = sinmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)

            if self.use_pro == "cosmax":
                scores = cosmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            if self.use_pro == "norm_cosmax":
                q_mul_k = map_norm(q_mul_k)
                scores = cosmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)

            if self.use_pro == "sin_2_max":
                q_mul_k = q_mul_k - 0.25 * np.pi
                scores = sin_2_max(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            if self.use_pro == "norm_sin_2_max":
                q_mul_k = map_norm(q_mul_k)
                q_mul_k = q_mul_k - 0.25 * np.pi
                scores = sin_2_max(q_mul_k / np.sqrt(self.d_per_head), dim=-1)

            if self.use_pro == "sin_2_max_move":
                scores = sin_2_max(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            if self.use_pro == "norm_sin_2_max_move":
                q_mul_k = map_norm(q_mul_k)
                scores = sin_2_max(q_mul_k / np.sqrt(self.d_per_head), dim=-1)

            if self.use_pro == "sirenmax":
                scores = sirenmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            if self.use_pro == "norm_sirenmax":
                q_mul_k = map_norm(q_mul_k)
                scores = sirenmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)

            if self.use_pro == "sin_softmax":
                q_mul_k = torch.sin(q_mul_k)
                scores = F.softmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            if self.use_pro == "norm_sin_softmax":
                q_mul_k = map_norm(q_mul_k)
                q_mul_k = torch.sin(q_mul_k)
                scores = F.softmax(q_mul_k / np.sqrt(self.d_per_head), dim=-1)
            scores_merge.append(scores)
            heads_v.append(torch.matmul(scores, v))
        attout = torch.cat(heads_v, dim=-1)
        attout = self.concat_heads(attout) + res
        res_mlp = attout
        attout = self.mlp_norm(attout)
        mlpout = self.mlp(attout)
        return mlpout + res_mlp, scores_merge


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=0):
        super(ConvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class MultiTrans(nn.Module):
    def __init__(self, d_model, h, num_class, depth=1, use_pro='softmax', mode='run'):
        super(MultiTrans, self).__init__()
        self.d_model = d_model
        self.h = h
        self.num_class = num_class
        self.use_pro = use_pro
        self.mode = mode
        self.convblocks = nn.Sequential(
            ConvBlock(3, int(self.d_model / 16), kernel_size=3, stride=2, padding=1),
            ConvBlock(int(self.d_model // 16), int(self.d_model // 4), kernel_size=3, stride=2, padding=1),
            ConvBlock(int(self.d_model // 4), int(self.d_model), kernel_size=3, stride=2, padding=1)
        )
        self.layers = []
        for i in range(depth):
            self.layers.append(TransformSingle(self.h, self.d_model, use_pro=use_pro, mode=mode))
        self.att = nn.Sequential(*self.layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.scores = nn.Linear(self.d_model, self.num_class)

    def forward(self, img):
        feature_map = self.convblocks(img).flatten(start_dim=2, end_dim=-1)
        att_feature = feature_map.transpose(2, 1)
        score_map_merge = []
        for layer in self.layers:
                att_feature, score_map = layer(att_feature)
                score_map_merge.append(score_map)
        avg_feature = self.pool(att_feature.transpose(2,1)).squeeze()
        scores = self.scores(avg_feature)
        return scores, score_map_merge


def check_accuracy_part34(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores, _ = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return acc

def attention_extract(test_image_dir, model):
    model.eval()
    with torch.no_grad():
        out_put = []
        transform = T.Compose([T.ToTensor(), 
                               T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        x = transform(Image.open(test_image_dir)).unsqueeze(0)
        x = x.to(device=device, dtype=dtype)
        _, attention_maps = model(x)
        for layer_attention in attention_maps:
          out_put.append([h_attention.cpu().numpy() for h_attention in layer_attention])
    return np.array(out_put)

def For_Train(model, optimizer, loader_train, loader_val, scheduler=None ,mode='search', epochs=1, test_image_dir=None):
    final_acc = 2
    model = model.to(device=device)
    accs = [0]
    train_acc = [0]
    losses = []
    attention_maps = []
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            scores, _ = model(x)
            _, pre = scores.max(1)
            num_correct = (pre == y).sum()
            num_samples = pre.size(0)
            train_acc.append(float(num_correct) / num_samples)
            loss = torch.nn.functional.cross_entropy(scores, y)
            loss_value = np.array(loss.item())
            losses.append(loss_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if t % 100 == 0:
                final_acc = check_accuracy_part34(loader_val, model)
                accs.append(np.array(final_acc))
                print(e, 'epoch', t, 'iter', 'Acc = ', final_acc)
        if mode == 'run' or mode == 'att_run':
            print("Epoch:", e)
        if e % 5 == 0 and mode == 'att_run':
            attention_maps.append(attention_extract(test_image_dir, model))
    return attention_maps, final_acc, accs, train_acc, losses

def run(mode='search', depth=1, use_pro='softmax',
        search_epoch=4, lr_range=[-5, -2], wd_range=[-4, -2],
        check_point_save_dir = None,
        run_epoch=30, lr=0, wd=0,  
        record_dir = None,
        dataset='cifar10', heads=8, d_model=512, lrsch=False, optim='Adam', test_image_dir=None):
    print(mode + 'ing for ' + use_pro)
    if mode == 'search':
        num_iter = 10000000
        epochs = search_epoch
    else:
        num_iter = 1
        epochs = run_epoch
    if dataset == 'cifar10':
        num_class = 10
    if dataset == 'cifar100':
        num_class = 100
    if dataset == 'tiny_imagenet':
        num_class = 200
    accs = None
    losses = None

    root = pro_root + 'dataset/' + dataset
    NUM_TRAIN = 49000
    transform = T.Compose([T.RandomRotation(90),
                        T.RandomHorizontalFlip(p=0.5),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    assert dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'tiny_imagenet'
    print('Loading ' + dataset + ' ......')
    if dataset == 'cifar10':
        cifar_train = dset.CIFAR10(root, train=True, download=True, transform=transform)
        loader_train = DataLoader(cifar_train, batch_size=32, sampler=sampler.SubsetRandomSampler(range(0, NUM_TRAIN)))

        cifar_val = dset.CIFAR10(root, train=True, download=True, transform=transform)
        loader_val = DataLoader(cifar_val, batch_size=512, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    if dataset == 'cifar100':
        cifar_train = dset.CIFAR100(root, train=True, download=True, transform=transform)
        loader_train = DataLoader(cifar_train, batch_size=32, sampler=sampler.SubsetRandomSampler(range(0, NUM_TRAIN)))

        cifar_val = dset.CIFAR100(root, train=True, download=True, transform=transform)
        loader_val = DataLoader(cifar_val, batch_size=512, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

    if dataset == 'tiny_imagenet':
        NUM_TRAIN = 95000

        cifar_train = get_imagenet(root, train=True, transform=transform)
        loader_train = DataLoader(cifar_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(0, NUM_TRAIN)))

        cifar_val = get_imagenet(root, train=True, transform=transform)
        loader_val = DataLoader(cifar_val, batch_size=512, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 99000)))


    print(dataset + ' Data loaded.')
    print('Optim: ' + optim)
    if lrsch:
        print('Lrsch used.')
    if mode == 'search':
        print('Searching under lr: (', lr_range[0], ',', lr_range[1],') , wd: (', wd_range[0], ',', wd_range[1], '), every ', epochs, ' epoch')
    if mode == 'run' or mode == 'att_run':
        print(mode + 'ing under lr: ' + str(lr) + ' , wd: ' + str(wd) + ' for ', str(epochs), ' epochs.')
    print('######################################     START     ######################################')
    for i in range(num_iter):
        model = MultiTrans(d_model, heads, num_class, depth=depth, use_pro=use_pro, mode=mode)
        if mode == 'search':
            learning_rate = 10 ** np.random.uniform(lr_range[0], lr_range[1])
            weight_decay = 10 ** np.random.uniform(wd_range[0], wd_range[1])
        else:
            learning_rate = lr 
            weight_decay = wd
        if optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=learning_rate,
                                        momentum=0.9,
                                        weight_decay=weight_decay,
                                        nesterov=True)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999), weight_decay=weight_decay)
        if lrsch :
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.2)
        else:
            scheduler = None
        attention_maps, val_acc, accs, train_acc, losses = For_Train(model, optimizer, loader_train, loader_val, scheduler=scheduler ,mode=mode, epochs=epochs, test_image_dir=test_image_dir)
        print(use_pro + ' noLrsc, '+ str(epochs) +' epoch, learning rate: ', learning_rate, ', weight decay: ', weight_decay, ', Val acc: ', val_acc)
        if mode == 'search':
            with open(check_point_save_dir, "a") as f:
                f.write(use_pro + ' noLrsch, ' + str(epochs) + ' epochs , learning rate:'  + str(learning_rate) + ', weight decay: ' + str(weight_decay) + ', Val acc: ' + str(val_acc) + '\n')
    if mode == 'run' or mode == 'att_run':
        record_dir_acc = record_dir + 'record_val_acc.npy'
        record_dir_train_acc = record_dir + 'record_train_acc.npy'
        record_dir_loss = record_dir + 'record_loss.npy'
        kq_value_dir = record_dir + 'kq_value.npy'

        np.save(record_dir_acc, np.array(accs))
        np.save(record_dir_train_acc, np.array(train_acc))
        np.save(record_dir_loss, np.array(losses))
        kq_value = np.vstack(kq_value)
        print(kq_value.shape)
        np.save(kq_value_dir, kq_value)
    if mode == 'att_run':
        record_dir_attention = record_dir + 'attention_maps.npy'
        np.save(record_dir_attention, attention_maps)


# Pro available:
    # softmax , norm_softmax
    # sinmax, norm_sinmax
    # cosmax, norm_cosmax
    # sin_2_max, norm_sin_2_max
    # sin_2_max_move, norm_sin_2_max_move
    # sirenmax, norm_sirenmax
    # sin_softmax, norm_sin_softmax

# Mode avilable: search, run, att_run
# dataset : cifar10, cifar100, tiny_imagenet

run(mode='att_run', depth=2, use_pro='norm_sin_2_max_move',
    # If for search:
    search_epoch=3, lr_range=[-5, -2], wd_range=[-4, -2],
    check_point_save_dir = pro_root + 'Attention_test/*functions/lr_wd_search.txt',
    # If for run:
    run_epoch=50, lr=0.00026021819225273693, wd=0.0001684884289701931,  ## norm_sin_2_max_move noLrsch, 4 epochs 0.256 set
    record_dir = pro_root + 'Attention_test/*functions/',
    # If for att_run
    test_image_dir = pro_root + 'att_test_sample.JPEG',  
    # Basic set:
    dataset='cifar100', heads=8, d_model=512, lrsch=False, optim='Adam')




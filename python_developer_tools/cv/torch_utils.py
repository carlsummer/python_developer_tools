import torch
import torchvision
import os
import random
import numpy as np


def init_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu

    # 设置随机种子  rank = -1
    # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，
    # 导致结果不确定。如果设置初始化，则每次初始化都是固定的
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False


def cuda2cpu(pred):
    # 将cuda的torch变量转为cpu
    if pred.is_cuda:
        pred_cpu = pred.cpu().numpy()
    else:
        pred_cpu = pred.numpy()
    return pred_cpu


def view_version_cuda_torch():
    """查看cuda cudnn torch 等版本是多少"""
    os.system('cat /usr/local/cuda/version.txt')
    os.system('cat /etc/issue')
    os.system('cat /proc/cpuinfo | grep name | sort | uniq')
    # os.system('whereis cudnn')
    try:
        head_file = open('/usr/local/cuda/include/cudnn.h')
    except:
        head_file = open('/usr/include/cudnn.h')
    lines = head_file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith('#define CUDNN_MAJOR'):
            line = line.split('#define CUDNN_MAJOR')
            n1 = int(line[1])
            continue
        if line.startswith('#define CUDNN_MINOR'):
            line = line.split('#define CUDNN_MINOR')
            n2 = int(line[1])
            continue
        if line.startswith('#define CUDNN_PATCHLEVEL'):
            line = line.split('#define CUDNN_PATCHLEVEL')
            n3 = int(line[1])
            break

    print("torch version", torch.__version__)
    print("torchvision version", torchvision.__version__)
    print("CUDA version", torch.version.cuda)
    print("CUDNN version", torch.backends.cudnn.version())
    print('CUDNN Version ', str(n1) + '.' + str(n2) + '.' + str(n3))


def select_device(device=''):
    """选择训练设备"""
    return torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    classes = labels  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount([labels[i]], minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights

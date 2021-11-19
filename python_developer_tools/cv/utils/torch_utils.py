import random
import importlib
import numpy as np
import os
import re
import subprocess
import sys
from collections import defaultdict
import PIL
import psutil
import torch
import torchvision
from tabulate import tabulate
from datetime import datetime
import torch.distributed as dist
from torchvision import transforms
from thop import profile
from copy import deepcopy

def get_model_info(model, tsize=(640,640)): # h,w
    """计算模型的参数量和计算一张图片的计算量"""
    stride = 64
    img = torch.zeros((1, 3, stride, stride), device=next(model.parameters()).device)
    flops, params = profile(deepcopy(model), inputs=(img,), verbose=False)
    params /= 1e6
    flops /= 1e9
    flops *= tsize[0] * tsize[1] / stride / stride * 2  # Gflops
    info = "Params: {:.6f}M, Gflops: {:.6f}".format(params, flops)
    return info

def recursive_to(input, device):
    """将输入的值转到设备cpu或者gpu中"""
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict):
        for name in input:
            if isinstance(input[name], torch.Tensor):
                input[name] = input[name].to(device)
        return input
    if isinstance(input, list):
        for i, item in enumerate(input):
            input[i] = recursive_to(item, device)
        return input
    assert False

def init_seeds(seed=0):
    """eg:init_seeds(seed=0)"""
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # gpu
    # 设置随机种子  rank = -1
    # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，
    # 导致结果不确定。如果设置初始化，则每次初始化都是固定的
    torch.manual_seed(seed) # cpu


def init_cudnn(reproducibility_training_speed=True):
    # https://blog.csdn.net/weixin_42587961/article/details/109363698
    # torch.backends.cudnn.deterministic将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法

    """
    reproducibility_training_speed为True表示训练速度加快，复现性;为false不能复现，提升网络性能
    https://blog.csdn.net/byron123456sfsfsfa/article/details/96003317
    设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，
    进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，
    其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
    对于卷积这个操作来说，其实现方式是多种多样的。最简单的实现方式就是使用多层循环嵌套，对于每张输入图像，对于每个要输出的通道，对于每个输入的通道，选取一个区域，
    同指定卷积核进行卷积操作，然后逐行滑动，直到整张图像都处理完毕，这个方法一般被称为 direct 法，
    这个方法虽然简单，但是看到这么多循环，我们就知道效率在一般情况下不会很高了。除此之外，实现卷积层的算法还有基于 GEMM (General Matrix Multiply) 的，
    基于 FFT 的，基于 Winograd 算法的等等，而且每个算法还有自己的一些变体。在一个开源的 C++ 库 triNNity 中，就实现了接近 80 种的卷积前向传播算法！
    每种卷积算法，都有其特有的一些优势，比如有的算法在卷积核大的情况下，速度很快；比如有的算法在某些情况下内存使用比较小。
    给定一个卷积神经网络（比如 ResNet-101），给定输入图片的尺寸，给定硬件平台，实现这个网络最简单的方法就是对所有卷积层都采用相同的卷积算法（比如 direct 算法），
    但是这样运行肯定不是最优的；比较好的方法是，我们可以预先进行一些简单的优化测试，在每一个卷积层中选择最适合（最快）它的卷积算法，决定好每层最快的算法之后，
    我们再运行整个网络，这样效率就会提升不少。
    """

    if reproducibility_training_speed:
        # 因此方便复现、提升训练速度就：
        torch.backends.cudnn.benchmark = False
        # 虽然通过torch.backends.cudnn.benchmark = False限制了算法的选择这种不确定性，但是由于，
        # 算法本身也具有不确定性，因此可以通过设置：
        torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    else:
        # 不需要复现结果、想尽可能提升网络性能：
        torch.backends.cudnn.benchmark = True



def tensor_to_PIL(tensor):
    # 输入tensor变量
    # 输出PIL格式图片
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

def tensor_to_np(tensor):
    #tensor转numpy
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

def cuda2cpu(pred):
    """
        将cuda的torch变量转为cpu
        eg:cuda2cpu(pred)"""
    if not hasattr(pred, 'is_cuda'):
        return pred
    if pred.is_cuda:
        pred_cpu = pred.cpu()
        if not hasattr(pred_cpu, 'detach'):
            pred_cpu = pred_cpu.numpy()
        else:
            pred_cpu = pred_cpu.detach().numpy()
    else:
        pred_cpu = pred.numpy()
    return pred_cpu


# -------------收集环境的版本等start-------------------#
def collect_torch_env():
    try:
        import torch.__config__

        return torch.__config__.show()
    except ImportError:
        # compatible with older versions of pytorch
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()


def getMemCpu():
    data = psutil.virtual_memory()
    total = data.total  # 总内存,单位为byte
    print('total', total)
    free = data.available  # 可用内存
    print('free', free)

    memory = "Memory usage:%d" % (int(round(data.percent))) + "%" + " "  # 内存使用情况
    print('memory', memory)
    cpu = "CPU:%0.2f" % psutil.cpu_percent(interval=1) + "%"  # CPU占用情况
    print('cpu', cpu)

def get_env_module():
    var_name = "DETECTRON2_ENV_MODULE"
    return var_name, os.environ.get(var_name, "<not set>")


def detect_compute_compatibility(CUDA_HOME, so_file):
    try:
        cuobjdump = os.path.join(CUDA_HOME, "bin", "cuobjdump")
        if os.path.isfile(cuobjdump):
            output = subprocess.check_output(
                "'{}' --list-elf '{}'".format(cuobjdump, so_file), shell=True
            )
            output = output.decode("utf-8").strip().split("\n")
            arch = []
            for line in output:
                line = re.findall(r"\.sm_([0-9]*)\.", line)[0]
                arch.append(".".join(line))
            arch = sorted(set(arch))
            return ", ".join(arch)
        else:
            return so_file + "; cannot find cuobjdump"
    except Exception:
        # unhandled failure
        return so_file


def collect_env_info():
    """查看cuda cudnn torch 等版本是多少"""
    has_gpu = torch.cuda.is_available()  # true for both CUDA & ROCM
    torch_version = torch.__version__

    # NOTE that CUDA_HOME/ROCM_HOME could be None even when CUDA runtime libs are functional
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    has_rocm = False
    if (getattr(torch.version, "hip", None) is not None) and (ROCM_HOME is not None):
        has_rocm = True
    has_cuda = has_gpu and (not has_rocm)

    data = []
    data.append(("sys.platform", sys.platform))  # check-template.yml depends on it
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    try:
        import detectron2  # noqa

        data.append(
            ("detectron2", detectron2.__version__ + " @" + os.path.dirname(detectron2.__file__))
        )
    except ImportError:
        data.append(("detectron2", "failed to import"))

    try:
        import detectron2._C as _C
    except ImportError as e:
        data.append(("detectron2._C", f"not built correctly: {e}"))

        # print system compilers when extension fails to build
        if sys.platform != "win32":  # don't know what to do for windows
            try:
                # this is how torch/utils/cpp_extensions.py choose compiler
                cxx = os.environ.get("CXX", "c++")
                cxx = subprocess.check_output("'{}' --version".format(cxx), shell=True)
                cxx = cxx.decode("utf-8").strip().split("\n")[0]
            except subprocess.SubprocessError:
                cxx = "Not found"
            data.append(("Compiler ($CXX)", cxx))

            if has_cuda and CUDA_HOME is not None:
                try:
                    nvcc = os.path.join(CUDA_HOME, "bin", "nvcc")
                    nvcc = subprocess.check_output("'{}' -V".format(nvcc), shell=True)
                    nvcc = nvcc.decode("utf-8").strip().split("\n")[-1]
                except subprocess.SubprocessError:
                    nvcc = "Not found"
                data.append(("CUDA compiler", nvcc))
    else:
        # print compilers that are used to build extension
        data.append(("Compiler", _C.get_compiler_version()))
        data.append(("CUDA compiler", _C.get_cuda_version()))  # cuda or hip
        if has_cuda and getattr(_C, "has_cuda", lambda: True)():
            data.append(
                ("detectron2 arch flags", detect_compute_compatibility(CUDA_HOME, _C.__file__))
            )

    data.append(get_env_module())
    data.append(("PyTorch", torch_version + " @" + os.path.dirname(torch.__file__)))
    data.append(("PyTorch debug build", torch.version.debug))

    data.append(("GPU available", has_gpu))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = ".".join((str(x) for x in torch.cuda.get_device_capability(k)))
            name = torch.cuda.get_device_name(k) + f" (arch={cap})"
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(("GPU " + ",".join(devids), name))

        if has_rocm:
            msg = " - invalid!" if not (ROCM_HOME and os.path.isdir(ROCM_HOME)) else ""
            data.append(("ROCM_HOME", str(ROCM_HOME) + msg))
        else:
            msg = " - invalid!" if not (CUDA_HOME and os.path.isdir(CUDA_HOME)) else ""
            data.append(("CUDA_HOME", str(CUDA_HOME) + msg))

            cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", None)
            if cuda_arch_list:
                data.append(("TORCH_CUDA_ARCH_LIST", cuda_arch_list))
    data.append(("Pillow", PIL.__version__))

    try:
        data.append(
            (
                "torchvision",
                str(torchvision.__version__) + " @" + os.path.dirname(torchvision.__file__),
            )
        )
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec("torchvision._C").origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(("torchvision arch flags", msg))
            except ImportError:
                data.append(("torchvision._C", "Not found"))
    except AttributeError:
        data.append(("torchvision", "unknown"))

    try:
        import fvcore

        data.append(("fvcore", fvcore.__version__))
    except ImportError:
        pass

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        data.append(("cv2", "Not found"))
    env_str = tabulate(data) + "\n"
    env_str += collect_torch_env()
    return env_str


# -------------收集环境的版本等end-------------------#

def view_version_cuda_torch():
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


def select_device(device='',batch_size=None):
    """选择训练设备
    eg:select_device("0")"""
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sCUDA:%g (%s, %dMB)" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        print(f'Using torch {torch.__version__} CPU')

    if cuda:
        if "," in device:
            return torch.device('cuda:0') #如果是多卡那么返回第一张卡
        else:
            return torch.device('cuda:{}'.format(device)) # 如果单卡并且是指定的卡号，那么直接返回
    else:
        return torch.device("cpu")

"""
torch.distributed.get_backend(group=group) # group是可选参数，返回字符串表示的后端 group表示的是ProcessGroup类
torch.distributed.get_rank(group=group) # group是可选参数，返回int，执行该脚本的进程的rank
torch.distributed.get_world_size(group=group) # group是可选参数,返回全局的整个的进程数
torch.distributed.is_initialized() # 判断该进程是否已经初始化
torch.distributed.is_mpi_avaiable() # 判断MPI是否可用
torch.distributed.is_nccl_avaiable() # 判断nccl是否可用
"""
def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


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


if __name__ == '__main__':
    # from python_developer_tools.cv.utils.torch_utils import collect_env_info
    print(collect_env_info())
    getMemCpu()

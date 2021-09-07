# [单机单卡](单机单卡.py)
```shell script
/home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 单机单卡.py
```

# [单机多卡](单机多卡.py)
```shell script
 /home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 -m torch.distributed.launch --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 29501 /home/zengxh/workspace/python_developer_tools/python_developer_tools/cv/train/不同数量卡训练/单机多卡.py
```

# 多机多卡
A机器运行
```shell script
sh train.sh 0 2
# 或者
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 /home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 -W ignore -m torch.distributed.launch --nproc_per_node 1 --node_rank 0 --nnodes 2 --master_addr 10.126.12.82 --master_port 29511 多机多卡.py
# 或者
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 /home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 -W ignore -m torch.distributed.launch --nproc_per_node 1 --node_rank 0 --nnodes 2 --master_addr 10.126.12.82 --master_port 29511 多机多卡.py
```
B机器运行
```shell script
sh train.sh 1 2
# 或者
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 /home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 -W ignore -m torch.distributed.launch --nproc_per_node 1 --node_rank 1 --nnodes 2 --master_addr 10.126.12.82 --master_port 29511 多机多卡.py
# 或者
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 /home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 -W ignore -m torch.distributed.launch --nproc_per_node 1 --node_rank 1 --nnodes 2 --master_addr 10.126.12.82 --master_port 29511 多机多卡.py
```
## 注意
- 两台机器的torch torchvision nccl 等库都要让这些电脑保持一致，我在实验过程中，一个使用torch1.8，一个使用1.7，导致两台机器跑不起来，之后都换成1.8之后才成功。
- 主机器的node_rank必须为0
## 异常
异常1
> RuntimeError: NCCL error in: /pytorch/torch/lib/c10d/ProcessGroupNCCL.cpp:825, 
>unhandled system error, NCCL version 2.7.8

解决方法1：下载安装
1. https://developer.nvidia.com/nccl/nccl-download
2. https://developer.nvidia.com/nccl/nccl-legacy-downloads
3. https://blog.csdn.net/m0_37426155/article/details/108129952
```shell script
rpm -i nccl-repo-rhel7-2.7.8-ga-cuda10.2-1-1.x86_64.rpm
yum install libnccl-2.7.8-1+cuda10.2 libnccl-devel-2.7.8-1+cuda10.2 libnccl-static-2.7.8-1+cuda10.2
```
解决方法2：
```shell script
git clone https://github.com/NVIDIA/nccl.git
cd nccl
sudo make install -j4
```
解决办法3
```shell script
conda activate CreepageDistance
conda install -y pytorch=1.8.1 torchvision torchaudio cudatoolkit=10.2 -c pytorch -c conda-forge
```
解决办法4：
```shell script
vim ~/.bashrc

export CUDA_HOME="/usr/local/cuda"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export NCCL_DEBUG=info
export NCCL_SOCKET_IFNAME=ens192
export NCCL_IB_DISABLE=1
export PATH="/home/zengxh/anaconda3/bin:$PATH"
. /home/zengxh/anaconda3/etc/profile.d/conda.sh

source ~/.bashrc
```

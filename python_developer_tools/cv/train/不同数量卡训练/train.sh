# m每台机器使用显卡数目
nproc_per_node=1
# 主机器ip
MASTER_ADDR=10.126.12.81
# 主机器端口号，可以随意，只要不冲突
MASTER_PORT=29511
# 机器编号，主机器必须为0
node_rank=$1
# 使用的机器数量
nnodes=$2
# 每个进程的线程数目
export OMP_NUM_THREADS=3
# 训练命令
DISTRIBUTED_ARGS="--nproc_per_node $nproc_per_node --node_rank $node_rank --nnodes $nnodes --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 /home/zengxh/anaconda3/envs/CreepageDistance/bin/python3.8 -W ignore -m torch.distributed.launch $DISTRIBUTED_ARGS main.py

#### tensorboard使用
import atexit
import os
import os.path as osp
import signal
import time

import torch
import torchvision
from tensorboardX import SummaryWriter
import subprocess
import sys

def run_tensorboard(out,port=6006):
    """创建并且运行"""
    board_out = osp.join(out, "tensorboard")
    if not osp.exists(board_out):
        os.makedirs(board_out)
    writer = SummaryWriter(board_out)

    tensorboard_bin_path = os.path.join(os.path.dirname(sys.executable), "tensorboard")
    cmdlist = [f"{tensorboard_bin_path}", f"--logdir={os.path.abspath(board_out)}", f"--port={port}","--host=0.0.0.0"]
    print(' '.join(cmdlist))
    p = subprocess.Popen(cmdlist)

    def killme():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(killme)

    return writer


if __name__ == '__main__':
    out="/home/zengxh/workspace/zdata"
    summarywriter=run_tensorboard(out,port=6006) # port为0的话是生成不冲突的随机端口

    # 记录模型
    model = torchvision.models.shufflenet_v2_x0_5()
    dummy_input = torch.randn(1, 3, 224, 224)
    summarywriter.add_graph(model, (dummy_input))

    # 记录训练时权重的值
    epoch = 10
    for name, param in model.named_parameters():
        summarywriter.add_histogram(name, param.clone().data, epoch)

    # 保存的是训练时候的图片
    train_inputs_make_grid = torchvision.utils.make_grid(dummy_input.to("cpu"), normalize=True,scale_each=True)
    summarywriter.add_image('Train Image{}'.format(epoch), train_inputs_make_grid,epoch)

    line1="this is ..."
    summarywriter.add_text('line1', str(line1), epoch)

    lr = 0.1
    for e in range(epoch):
        lr = lr * 0.1
        summarywriter.add_scalar("train/lr", lr, global_step=e)
        summarywriter.add_scalar("train/lr2", lr, global_step=e)
        summarywriter.add_scalar("val/lr2", lr, global_step=e)

    summarywriter.close()  # 关闭tensorboardX 日志

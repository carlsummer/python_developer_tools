#### tensorboard使用
```python
import atexit
import os
import os.path as osp
import signal
def run_tensorboard(self):
    board_out = osp.join(self.out, "tensorboard")
    if not osp.exists(board_out):
        os.makedirs(board_out)
    self.writer = SummaryWriter(board_out)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    mkdir(os.path.abspath(board_out))
    p = subprocess.Popen(
        ["/home/zengxh/anaconda3/envs/CreepageDistance/bin/tensorboard", f"--logdir={os.path.abspath(board_out)}", f"--port={C.io.tensorboard_port}","--host=0.0.0.0"]
    )

    def killme():
        os.kill(p.pid, signal.SIGTERM)

    atexit.register(killme)
```
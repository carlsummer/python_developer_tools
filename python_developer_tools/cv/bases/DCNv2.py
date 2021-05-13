# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:5/12/2021 9:00 PM
# @File:DcnV2
"""
conda create -n DCNV2 python=3.8
conda activate DCNV2
git clone https://github.com/jinfagang/DCNv2_latest.git
cd DCNv2_latest/
pip install torch==1.6.0
pip install torchvision==0.7.0
python3 setup.py build develop

./make.sh         # build
/home/deploy/anaconda3/envs/yolov5_py38_cu102_conda/lib/python3.8/site-packages/torch/utils/cpp_extension.py
如果报错subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
那么将命令command = ['ninja', '-v']改为command = ['ninja', '-V']

如果报错：
g++: error: /home/deploy/software/DCNv2-master/build/temp.linux-x86_64-3.8/home/deploy/software/DCNv2-master/src/vision.o: No such file or directory
g++: error: /home/deploy/software/DCNv2-master/build/temp.linux-x86_64-3.8/home/deploy/software/DCNv2-master/src/cpu/dcn_v2_cpu.o: No such file or directory
g++: error: /home/deploy/software/DCNv2-master/build/temp.linux-x86_64-3.8/home/deploy/software/DCNv2-master/src/cpu/dcn_v2_im2col_cpu.o: No such file or directory
g++: error: /home/deploy/software/DCNv2-master/build/temp.linux-x86_64-3.8/home/deploy/software/DCNv2-master/src/cpu/dcn_v2_psroi_pooling_cpu.o: No such file or directory
g++: error: /home/deploy/software/DCNv2-master/build/temp.linux-x86_64-3.8/home/deploy/software/DCNv2-master/src/cuda/dcn_v2_cuda.o: No such file or directory
g++: error: /home/deploy/software/DCNv2-master/build/temp.linux-x86_64-3.8/home/deploy/software/DCNv2-master/src/cuda/dcn_v2_im2col_cuda.o: No such file or directory
g++: error: /home/deploy/software/DCNv2-master/build/temp.linux-x86_64-3.8/home/deploy/software/DCNv2-master/src/cuda/dcn_v2_psroi_pooling_cuda.o: No such file or directory
那么：
python3 setup.py build develop

python testcpu.py    # run examples and gradient check on cpu
python testcuda.py   # run examples and gradient check on gpu
"""

# An Example
# deformable conv
import torch
from dcn_v2 import DCN
input = torch.randn(2, 64, 128, 128).cuda()
# wrap all things (offset and mask) in DCN
dcn = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, deformable_groups=2).cuda()
output = dcn(input)
print(output.shape)

# deformable roi pooling
from dcn_v2 import DCNPooling
input = torch.randn(2, 32, 64, 64).cuda()
batch_inds = torch.randint(2, (20, 1)).cuda().float()
x = torch.randint(256, (20, 1)).cuda().float()
y = torch.randint(256, (20, 1)).cuda().float()
w = torch.randint(64, (20, 1)).cuda().float()
h = torch.randint(64, (20, 1)).cuda().float()
rois = torch.cat((batch_inds, x, y, x + w, y + h), dim=1)

# mdformable pooling (V2)
# wrap all things (offset and mask) in DCNPooling
dpooling = DCNPooling(spatial_scale=1.0 / 4,
                     pooled_size=7,
                     output_dim=32,
                     no_trans=False,
                     group_size=1,
                     trans_std=0.1).cuda()

dout = dpooling(input, rois)
print(dout.shape)
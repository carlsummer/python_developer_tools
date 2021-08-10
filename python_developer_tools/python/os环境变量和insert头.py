# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:8/4/2021 9:22 AM
# @File:os环境变量和insert头
import os
os.environ["CUDA_HOME"]="/usr/local/cuda-10.2"
os.environ["TENSORRT_HOME"]="/TensorRT-7.2.2.3"
os.environ["LD_LIBRARY_PATH"]="/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
os.environ["PATH"]='/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3'
import sys
sys.path.append('..') #表示导入当前文件的上层目录到搜索路径中
sys.path.append('/home/model') # 绝对路径
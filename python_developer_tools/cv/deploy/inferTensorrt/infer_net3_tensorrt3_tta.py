# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:10 AM
# @File:infer_net3_tensorrt3_tta.py
import os
os.environ["CUDA_HOME"]="/usr/local/cuda-10.2"
os.environ["TENSORRT_HOME"]="/TensorRT-7.2.2.3"
os.environ["LD_LIBRARY_PATH"]="/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/TensorRT-7.2.2.3/lib:/usr/local/cuda-10.2/lib:/usr/local/cuda-10.2/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
os.environ["PATH"]='/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3:/usr/local/cuda-10.2/bin:/TensorRT-7.2.2.3'
from ai_hub import inferServer
import json
from io import BytesIO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.tools import make_default_context
cuda.init()  # Initialize CUDA
ctx = make_default_context()  # Create CUDA context
import numpy as np
import cv2
from PIL import Image
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.nn import functional as F
import torch

class myInfer(inferServer):
    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()  # pycuda 操作缓冲区
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)  # 分配内存
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        self.cfx.push()

        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]  # 将输入放入device
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)  # 执行模型推理
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]  # 将预测结果从缓冲区取出
        stream.synchronize()  # 线程同步
        return [out.host for out in outputs]

    def get_engine(self, engine_file_path=""):
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _create_context(self):
        return self.engine.create_execution_context()

    def __init__(self, model=None):
        super().__init__(model)
        self.cfx = cuda.Device(0).make_context()
        self.trt_logger = trt.Logger()

        engine_file_path = '/user_data/model_data/checkpoint-best.engine'
        self.engine = self.get_engine(engine_file_path)
        self.context = self._create_context()

        self.inputs, self.outputs, self.bindings, self.stream = \
            self.allocate_buffers(self.engine)


    # 数据前处理
    def pre_process(self, data):
        json_data = json.loads(data.get_data().decode("utf-8"))
        img = json_data.get("img")
        bast64_data = img.encode(encoding="utf-8")
        img = base64.b64decode(bast64_data)
        img = cv2.imdecode(np.fromstring(img, dtype=np.uint8), -1)
        # img = np.expand_dims(img, 0).astype(np.float16)
        return img

    # 数据后处理
    def post_process(self, data):
        # data = data.squeeze().data.numpy()
        # data = data[0].reshape(10,256, 256)
        # data = np.argmax(data, axis=0)
        # data = data.squeeze()
        # data = data.squeeze().cpu().data.numpy()
        # data = data+1
        data[data == 0]=1
        img_encode = np.array(cv2.imencode('.png', data)[1]).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data, 'utf-8')
        return bast64_str

    # 模型预测：默认执行self.model(preprocess_data)，一般不用重写
    # 如需自定义，可覆盖重写
    def predict(self, data):
        np.copyto(self.inputs[0].host, data.ravel())
        trt_outputs = self.do_inference(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        self.cfx.pop()
        data = trt_outputs[0].reshape(256, 256)
        return data

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        super(HostDeviceMem, self).__init__()
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

if __name__ == "__main__":
    # net1
    # model_name = 'resnet50'#'efficientnet-b6'
    # n_class=10
    # net1=seg_qyl(model_name,n_class)
    # # net1 = torch.nn.DataParallel(net1) # 如果使用了多卡训练那么需要加这句话
    # checkpoints=torch.load('/user_data/model_data/checkpoint-best.pth')
    # net1.load_state_dict(checkpoints['state_dict'])
    # net1.cuda()
    # net1.eval()

    my_infer = myInfer()
    my_infer.run(ip="0.0.0.0",debuge=False)  # 默认为("127.0.0.1", 8080)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息

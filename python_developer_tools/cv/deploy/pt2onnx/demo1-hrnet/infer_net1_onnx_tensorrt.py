# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/26/2021 10:12 AM
# @File:infer_net1_onnx.py
from ai_hub import inferServer
import json
import base64
import cv2
from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch.autograd import Variable as V
import base64
import onnx
import onnx_tensorrt.backend as backend
from ours_code.net1.infer import get_infer_transform
from ours_code.net1.models.model import seg_qyl


class myInfer(inferServer):
    def __init__(self, model):
        super().__init__(model)

    # 数据前处理
    def pre_process(self, data):
        # json process
        json_data = json.loads(data.get_data().decode('utf-8'))
        img = json_data.get("img")
        bast64_data = img.encode(encoding='utf-8')
        img = base64.b64decode(bast64_data)
        bytesIO = BytesIO()
        img = Image.open(BytesIO(bytearray(img)))
        img = np.array(img)
        img = img.astype(np.float32)

        #net1
        image_cv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_cv = cv2.resize(image_cv, (256, 256))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = np.array(image_cv, dtype=float) / 255.
        img_np = (img_np - mean) / std
        img_np = img_np.transpose((2, 0, 1))
        img_np_nchw = np.expand_dims(img_np, 0).astype(np.float16)
        img_np_nchw = np.array(img_np_nchw, dtype=img_np_nchw.dtype, order='C')
        return img_np_nchw

    # 数据后处理
    def post_process(self, data):
        data = np.argmax(data, axis=1)
        data = data.squeeze()
        # data = data.squeeze().cpu().data.numpy()
        data = data+1

        # data = data.astype(np.uint8)
        # data = Image.fromarray(data)
        # data = data.convert('L')
        # data = data.resize((256, 256), resample=Image.NEAREST)
        # data = np.array(data)

        img_encode = np.array(cv2.imencode('.png', data)[1]).tobytes()
        bast64_data = base64.b64encode(img_encode)
        bast64_str = str(bast64_data, 'utf-8')
        return bast64_str

    # 模型预测：默认执行self.model(preprocess_data)，一般不用重写
    # 如需自定义，可覆盖重写
    def predict(self, data):
      ret = self.model.run(data)[0]
      return ret


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

    model = onnx.load("/user_data/model_data/checkpoint-best.onnx")
    engine = backend.prepare(model, device='CUDA:0')

    my_infer = myInfer(engine)
    my_infer.run(ip="0.0.0.0",debuge=False)  # 默认为("127.0.0.1", 8080)，可自定义端口，如用于天池大赛请默认即可，指定debuge=True可获得更多报错信息

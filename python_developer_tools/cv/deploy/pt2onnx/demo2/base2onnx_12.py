# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/28/2021 5:17 PM
# @File:base2onnx
import torch
import onnxruntime  # cuda10.2==onnxruntime-gpu 1.5.2
import torch.nn.functional as F


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        ap = F.max_pool2d(x.unsqueeze(0), 3, stride=1, padding=1)
        ap = ap.squeeze(0)
        mask = (x == ap).float().clamp(min=0.0)
        return x * mask


net = JustReshape()
model_name = 'just_reshape.onnx'
dummy_input = torch.randn(1, 128, 16).cuda()
torch.onnx.export(net, dummy_input, model_name,
                  opset_version=11,
                  verbose=True,
                  input_names=['input'],
                  output_names=['output'])

ort_session = onnxruntime.InferenceSession(model_name)

input_name = ort_session.get_inputs()[0].name
label_name = ort_session.get_outputs()[0].name

nlines = ort_session.run([label_name], {input_name: dummy_input.cpu().numpy(), })
print(nlines)

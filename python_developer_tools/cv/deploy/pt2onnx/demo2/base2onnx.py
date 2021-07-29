# !/usr/bin/env python
# -- coding: utf-8 --
# @Author zengxiaohui
# Datatime:7/28/2021 5:17 PM
# @File:base2onnx
import torch
import onnxruntime  #cuda10.2==onnxruntime-gpu 1.5.2


class JustReshape(torch.nn.Module):
    def __init__(self):
        super(JustReshape, self).__init__()

    def forward(self, x):
        return x.view((x.shape[0], x.shape[1], x.shape[3], x.shape[2]))


net = JustReshape()
model_name = 'just_reshape.onnx'
dummy_input = torch.randn(2, 3, 4, 5)
torch.onnx.export(net, dummy_input, model_name, input_names=['input'], output_names=['output'])

ort_session = onnxruntime.InferenceSession(model_name)

input_name = ort_session.get_inputs()[0].name
label_name = ort_session.get_outputs()[0].name

nlines = ort_session.run([label_name], { input_name:dummy_input.numpy(), })
print(nlines)




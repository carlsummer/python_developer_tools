import torch.nn as nn
import torch

def make_stem_layer(in_channels, stem_channels):
    """Make stem layer for ResNet. self.deep_stem:"""
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            stem_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False),
        nn.BatchNorm2d(stem_channels // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            stem_channels // 2,
            stem_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False),
        nn.BatchNorm2d(stem_channels // 2),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            stem_channels // 2,
            stem_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True))

if __name__ == '__main__':
    model = make_stem_layer(3,64)
    input = torch.randn(128, 3, 32, 32)
    out = model(input)
    print(out.shape)
    torch.onnx.export(model, input, "xx.onnx", input_names=['input'], output_names=['output'])
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

def SA_init(input_dim, output_dim, sa_num, r1=0.5, r2=1.0):
    # input_dim: input data dimension;  output_dim: output / last layer feature dimension
    # input_dim * s(sigmoid(alpha)) ^ sa_num = output_dim, find alpha
    # s(sigmoid(alpha)) = r1 + (r2 - r1) * sigmoid(alpha)
    eps = 1e-4  # avoid inf
    if input_dim * r1 ** sa_num > output_dim:
        return np.log(eps)
    else:
        return np.log(-(np.power(output_dim / input_dim, 1.0/sa_num) - r1) / (np.power(output_dim / input_dim, 1.0/sa_num) - r2) + eps)

def shape_adaptor(poollayer,input, alpha):
    sigmoid_alpha = torch.sigmoid(alpha)
    s_alpha = 0.5 * sigmoid_alpha.item() + 0.5

    # use local-type shape adaptors
    input1_rs = F.interpolate(poollayer(input), scale_factor=2 * s_alpha, mode='bilinear', align_corners=True)
    input2_rs = F.interpolate(input, size=input1_rs.shape[-2:], mode='bilinear', align_corners=True)

    return (1 - sigmoid_alpha) * input1_rs + sigmoid_alpha * input2_rs

if __name__ == '__main__':
    # initialisations computed by; d_in = 32 and d_out = 8
    alpha = nn.Parameter(-0.346 * torch.ones(1, requires_grad=True)) # 有多少层maxpool，torch.ones里面就设置多少
    pooling = nn.MaxPool2d(2, 2)
    x = torch.randn(1, 16, 64, 64)
    x = shape_adaptor(pooling, x, alpha[0])
    print(x.shape)
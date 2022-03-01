# csdn: https://blog.csdn.net/Carlsummer/article/details/123186276
# github:
import torch

def sharpen(p, T):
    """
    sharpen: eq. (7), algorithm 1 line 8
    :param p: post disatribution: [N,10]
    [[0.22,0.32........], =>sum =1
    [0.01,0.3,0.03.....], =>sum =1
    ...]
    [0.1,0.1,0.1,.... 0.2,0.1]
    :param T: temperature
    :return: sharpened result
    """
    p_power = torch.pow(p, 1. / T)
    return p_power / torch.sum(p_power, dim=-1, keepdim=True)
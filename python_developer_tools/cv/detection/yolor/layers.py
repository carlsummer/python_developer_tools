from torch import nn
import torch


class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit


class ImplicitC(nn.Module):
    def __init__(self, channel):
        super(ImplicitC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit


class ShiftChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ShiftChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) + x


class ControlChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(ControlChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return a.expand_as(x) * x


class AlternateChannel(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    def __init__(self, layers):
        super(AlternateChannel, self).__init__()
        self.layers = layers  # layer indices

    def forward(self, x, outputs):
        a = outputs[self.layers[0]]
        return torch.cat([a.expand_as(x), x], dim=1)


if __name__ == '__main__':
    # iPR
    x = torch.randn(1, 128, 80, 80)  # backone的输出
    layer1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 这里指的是Conv BN Act 另外2个没写
    x = layer1(x)

    iA = ImplicitA(256)
    ia = iA()
    x = ia.expand_as(x) + x

    layer2 = nn.Conv2d(256, 21, kernel_size=3, padding=1)
    x = layer2(x)

    iM = ImplicitM(21)
    im = iM()
    x = im.expand_as(x) * x
    print(x.shape)

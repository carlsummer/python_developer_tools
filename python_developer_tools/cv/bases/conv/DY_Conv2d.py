
import torch
import torch.nn as nn
import torch.nn.functional as F


class DY_Conv2d(nn.Conv2d):

    def __init__(self, in_chan, out_chan, kernel_size=3,
            stride=1, padding=1, dilation=1, groups=1, bias=False,
            act=nn.ReLU(inplace=True), K=4,
            temperature=30, temp_anneal_steps=3000):
        super(DY_Conv2d, self).__init__(
            in_chan, out_chan * K, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)
        assert in_chan // 4 > 0
        self.K = K
        self.act = act
        self.se_conv1 = nn.Conv2d(in_chan, in_chan // 4, 1, 1, 0, bias=True)
        self.se_conv2 = nn.Conv2d(in_chan // 4, K, 1, 1, 0, bias=True)
        self.temperature = temperature
        self.temp_anneal_steps = temp_anneal_steps
        self.temp_interval = (temperature - 1) / temp_anneal_steps

    def get_atten(self, x):
        bs, _, h, w = x.size()
        atten = torch.mean(x, dim=(2, 3), keepdim=True)
        atten = self.se_conv1(atten)
        atten = self.act(atten)
        atten = self.se_conv2(atten)
        if self.training and self.temp_anneal_steps > 0:
            atten = atten / self.temperature
            self.temperature -= self.temp_interval
            self.temp_anneal_steps -= 1
        atten = atten.softmax(dim=1).view(bs, -1)
        return atten


    def forward(self, x):
        bs, _, h, w = x.size()
        atten = self.get_atten(x)

        out_chan, in_chan, k1, k2 = self.weight.size()
        W = self.weight.view(1, self.K, -1, in_chan, k1, k2)
        W = (W * atten.view(bs, self.K, 1, 1, 1, 1)).sum(dim=1)
        W = W.view(-1, in_chan, k1, k2)

        b = self.bias
        if not b is None:
            b = b.view(1, self.K, -1)
            b = (b * atten.view(bs, self.K, 1)).sum(dim=1).view(-1)

        x = x.view(1, -1, h, w)

        out = F.conv2d(x, W, b, self.stride, self.padding,
                self.dilation, self.groups * bs)
        out = out.view(bs, -1, out.size(2), out.size(3))
        return out


if __name__ == '__main__':
    net = DY_Conv2d(32, 64, 3, 1, 1, bias=True)
    inten = torch.randn(2, 32, 224, 224)
    out = net(inten)
    print(out.size())

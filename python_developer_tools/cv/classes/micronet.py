import math
from python_developer_tools.cv.bases.conv.DepthSpatialSepConvs import SpatialSepConvSF
from python_developer_tools.cv.bases.pool.AvgPool2d import SwishAdaptiveAvgPool2d
from python_developer_tools.cv.bases.pool.MaxGroupPooling import MaxGroupPooling
from python_developer_tools.cv.models.blocks.DYMicroBlock import *
from python_developer_tools.cv.models.head.classes.MicroNet import MicroNet_head

msnx_dy6_exp4_4M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4,y1,y2,y3,r
        [2, 1,   8, 3, 2, 2,  0,  4,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 4,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 4,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 4,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 384, 3, 1, 4, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]
msnx_dy6_exp6_6M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1,   8, 3, 2, 2,  0,  6,   8,  2,  2, 2, 0, 1, 1],  #6->12(0, 0)->24  ->8(4,2)->8
        [2, 1,  16, 3, 2, 2,  0,  8,  16,  4,  4, 2, 2, 1, 1], #8->16(0, 0)->32  ->16(4,4)->16
        [2, 1,  16, 5, 2, 2,  0, 16,  16,  4,  4, 2, 2, 1, 1], #16->32(0, 0)->64  ->16(8,2)->16
        [1, 1,  32, 5, 1, 6,  4,  4,  32,  4,  4, 2, 2, 1, 1], #16->16(2,8)->96 ->32(8,4)->32
        [2, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 1], #32->32(2,16)->192 ->64(12,4)->64
        [1, 1,  96, 3, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(3,16)->384 ->96(16,6)->96
        [1, 1, 576, 3, 1, 6, 12, 12,   0,  0,  0, 2, 2, 1, 2], #96->96(4,24)->576
]
msnx_dy9_exp6_12M_221_cfgs = [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1,  12, 3, 2, 2,  0,  8,  12,  4,  4, 2, 0, 1, 1], #8->16(0, 0)->32  ->12(4,3)->12
        [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 2, 2, 1, 1], #12->24(0,0)->48  ->16(8, 2)->16
        [1, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 2, 2, 1, 1], #16->16(0, 0)->64  ->24(8,3)->24
        [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 2, 2, 1, 1], #24->24(2, 12)->144  ->32(16,2)->32
        [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 2, 2, 1, 2], #32->32(2,16)->192 ->32(16,2)->32
        [1, 1,  64, 5, 1, 6,  8,  8,  64,  8,  8, 2, 2, 1, 2], #32->32(2,16)->192 ->64(12,4)->64
        [2, 1,  96, 5, 1, 6,  8,  8,  96,  8,  8, 2, 2, 1, 2], #64->64(4,12)->384 ->96(16,5)->96
        [1, 1, 128, 3, 1, 6, 12, 12, 128,  8,  8, 2, 2, 1, 2], #96->96(5,16)->576->128(16,8)->128
        [1, 1, 768, 3, 1, 6, 16, 16,   0,  0,  0, 2, 2, 1, 2], #128->128(4,32)->768
        ]
msnx_dy12_exp6_20M_020_cfgs = [
    #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
    [2, 1,  16, 3, 2, 2,  0, 12,  16,  4,  4, 0, 2, 0, 1], #12->24(0, 0)->48  ->16(8,2)->16
    [2, 1,  24, 3, 2, 2,  0, 16,  24,  4,  4, 0, 2, 0, 1], #16->32(0, 0)->64  ->24(8,3)->24
    [1, 1,  24, 3, 2, 2,  0, 24,  24,  4,  4, 0, 2, 0, 1], #24->48(0, 0)->96  ->24(8,3)->24
    [2, 1,  32, 5, 1, 6,  6,  6,  32,  4,  4, 0, 2, 0, 1], #24->24(2,12)->144  ->32(16,2)->32
    [1, 1,  32, 5, 1, 6,  8,  8,  32,  4,  4, 0, 2, 0, 2], #32->32(2,16)->192 ->32(16,2)->32
    [1, 1,  64, 5, 1, 6,  8,  8,  48,  8,  8, 0, 2, 0, 2], #32->32(2,16)->192 ->48(12,4)->48
    [1, 1,  80, 5, 1, 6,  8,  8,  80,  8,  8, 0, 2, 0, 2], #48->48(3,16)->288 ->80(16,5)->80
    [1, 1,  80, 5, 1, 6, 10, 10,  80,  8,  8, 0, 2, 0, 2], #80->80(4,20)->480->80(20,4)->80
    [2, 1, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0, 2], #80->80(4,20)->480->128(16,8)->128
    [1, 1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0, 2], #120->128(4,32)->720->128(32,4)->120
    [1, 1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0, 2], #120->128(4,32)->720->160(32,5)->144
    [1, 1, 864, 3, 1, 6, 12, 12,   0,  0,  0, 0, 2, 0, 2], #144->144(5,32)->864
]

def get_micronet_config(mode):
    return eval(mode+'_cfgs')

def conv_3x3_bn(inp, oup, stride, dilation=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False, dilation=dilation),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def gcd(a, b):
    a, b = (a, b) if a >= b else (b, a)
    while b:
        a, b = b, a%b
    return a

class StemLayer(nn.Module):
    def __init__(self, inp, oup, stride, dilation=1, mode='default', groups=(4,4)):
        super(StemLayer, self).__init__()

        self.exp = 1 if mode == 'default' else 2
        g1, g2 = groups 
        if mode == 'default':
            self.stem = nn.Sequential(
                nn.Conv2d(inp, oup*self.exp, 3, stride, 1, bias=False, dilation=dilation),
                nn.BatchNorm2d(oup*self.exp),
                nn.ReLU6(inplace=True) if self.exp == 1 else MaxGroupPooling(self.exp)
            )
        elif mode == 'spatialsepsf':
            self.stem = nn.Sequential(
                SpatialSepConvSF(inp, groups, 3, stride),
                MaxGroupPooling(2) if g1*g2==2*oup else nn.ReLU6(inplace=True)
            )
        else: 
            raise ValueError('Undefined stem layer')
           
    def forward(self, x):
        out = self.stem(x)    
        return out


class MicroNet(nn.Module):
    def __init__(self, cfg, num_classes=1000, teacher=False):
        super(MicroNet, self).__init__()

        mode = cfg.MODEL.MICRONETS.NET_CONFIG
        self.cfgs = get_micronet_config(mode)

        block = eval(cfg.MODEL.MICRONETS.BLOCK)
        stem_mode = cfg.MODEL.MICRONETS.STEM_MODE
        stem_ch = cfg.MODEL.MICRONETS.STEM_CH
        stem_dilation = cfg.MODEL.MICRONETS.STEM_DILATION
        stem_groups = cfg.MODEL.MICRONETS.STEM_GROUPS
        out_ch = cfg.MODEL.MICRONETS.OUT_CH
        depthsep = cfg.MODEL.MICRONETS.DEPTHSEP
        shuffle = cfg.MODEL.MICRONETS.SHUFFLE
        pointwise = cfg.MODEL.MICRONETS.POINTWISE
        dropout_rate = cfg.MODEL.MICRONETS.DROPOUT

        act_max = cfg.MODEL.ACTIVATION.ACT_MAX
        act_bias = cfg.MODEL.ACTIVATION.LINEARSE_BIAS
        activation_cfg= cfg.MODEL.ACTIVATION

        # building first layer
        input_channel = stem_ch
        layers = [StemLayer(
                    3, input_channel,
                    stride=2, 
                    dilation=stem_dilation, 
                    mode=stem_mode,
                    groups=stem_groups
                )]

        for idx, val in enumerate(self.cfgs):
            s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r = val

            t1 = (c1, c2)
            gs1 = (g1, g2)
            gs2 = (c3, g3, g4)
            activation_cfg.dy = [y1, y2, y3]
            activation_cfg.ratio = r

            output_channel = c
            layers.append(block(input_channel, output_channel,
                kernel_size=ks, 
                stride=s, 
                ch_exp=t1, 
                ch_per_group=gs1, 
                groups_1x1=gs2,
                depthsep = depthsep,
                shuffle = shuffle,
                pointwise = pointwise,
                activation_cfg=activation_cfg,
            ))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 
                    kernel_size=ks, 
                    stride=1, 
                    ch_exp=t1, 
                    ch_per_group=gs1, 
                    groups_1x1=gs2,
                    depthsep = depthsep,
                    shuffle = shuffle,
                    pointwise = pointwise,
                    activation_cfg=activation_cfg,
                ))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        self.avgpool = SwishAdaptiveAvgPool2d()

        # building last several layers
        output_channel = out_ch
         
        self.classifier = MicroNet_head(input_channel,output_channel,dropout_rate,num_classes)
        self._initialize_weights()
           
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

def get_cfg():
    cfg = CN()
    cfg.MODEL = CN()
    cfg.MODEL.ACTIVATION = CN()
    cfg.MODEL.MICRONETS = CN()
    cfg.MODEL.MICRONETS.POINTWISE = 'group'  # fft/1x1/shuffle
    cfg.MODEL.MICRONETS.BLOCK = "DYMicroBlock"
    cfg.MODEL.MICRONETS.STEM_DILATION = 1
    cfg.MODEL.MICRONETS.STEM_MODE = "spatialsepsf"
    cfg.MODEL.MICRONETS.DEPTHSEP = True
    cfg.MODEL.ACTIVATION.MODULE = "DYShiftMax"
    cfg.MODEL.ACTIVATION.ACT_MAX = 2.0
    cfg.MODEL.ACTIVATION.LINEARSE_BIAS = False
    cfg.MODEL.ACTIVATION.INIT_A_BLOCK3 = 1.0, 0.0
    cfg.MODEL.ACTIVATION.REDUCTION = 8
    cfg.MODEL.MICRONETS.SHUFFLE = True
    return cfg

def MicroNet_msnx_dy6_exp4_4M_221(num_classes=1000) -> MicroNet:
    cfg=get_cfg()
    cfg.MODEL.MICRONETS.NET_CONFIG = "msnx_dy6_exp4_4M_221"
    cfg.MODEL.MICRONETS.STEM_CH = 4
    cfg.MODEL.MICRONETS.STEM_GROUPS = 2, 2
    cfg.MODEL.MICRONETS.OUT_CH = 640
    cfg.MODEL.MICRONETS.DROPOUT = 0.05
    cfg.MODEL.ACTIVATION.INIT_A = 1.0, 1.0
    cfg.MODEL.ACTIVATION.INIT_B = 0.0, 0.0
    return MicroNet(cfg,num_classes)

def MicroNet_msnx_dy6_exp6_6M_221(num_classes=1000) -> MicroNet:
    cfg = get_cfg()
    cfg.MODEL.MICRONETS.NET_CONFIG = "msnx_dy6_exp6_6M_221"
    cfg.MODEL.MICRONETS.STEM_CH = 6
    cfg.MODEL.MICRONETS.STEM_GROUPS = 3, 2
    cfg.MODEL.MICRONETS.OUT_CH = 960
    cfg.MODEL.MICRONETS.DROPOUT = 0.05
    cfg.MODEL.ACTIVATION.INIT_A = 1.0, 1.0
    cfg.MODEL.ACTIVATION.INIT_B = 0.0, 0.0
    return MicroNet(cfg,num_classes)

def MicroNet_msnx_dy9_exp6_12M_221(num_classes=1000) -> MicroNet:
    cfg = get_cfg()
    cfg.MODEL.MICRONETS.NET_CONFIG = "msnx_dy9_exp6_12M_221"
    cfg.MODEL.MICRONETS.STEM_CH = 8
    cfg.MODEL.MICRONETS.STEM_GROUPS = 4, 2
    cfg.MODEL.MICRONETS.OUT_CH = 1024
    cfg.MODEL.MICRONETS.DROPOUT = 0.1
    cfg.MODEL.ACTIVATION.INIT_A = 1.0, 1.0
    cfg.MODEL.ACTIVATION.INIT_B = 0.0, 0.0
    return MicroNet(cfg,num_classes)

def MicroNet_msnx_dy12_exp6_20M_020(num_classes=1000) -> MicroNet:
    cfg = get_cfg()
    cfg.MODEL.MICRONETS.NET_CONFIG = "msnx_dy12_exp6_20M_020"
    cfg.MODEL.MICRONETS.STEM_CH = 12
    cfg.MODEL.MICRONETS.STEM_GROUPS = 4, 3
    cfg.MODEL.MICRONETS.OUT_CH = 1024
    cfg.MODEL.MICRONETS.DROPOUT = 0.1
    cfg.MODEL.ACTIVATION.INIT_A = 1.0, 0.5
    cfg.MODEL.ACTIVATION.INIT_B = 0.0, 0.5
    return MicroNet(cfg,num_classes)

if __name__ == '__main__':
    model = MicroNet_msnx_dy6_exp4_4M_221()
    input = torch.randn(64, 3, 125, 476)
    out = model(input)
    print(out.shape)
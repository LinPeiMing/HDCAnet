# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
HDCAnet
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.arch_util import MySequential, LayerNorm2d, ResBlock, count_parameters
from basicsr.archs.local_arch import Local_Base
from basicsr.utils.registry import ARCH_REGISTRY
from moco.builder import MoCo
from mmcv.ops import modulated_deform_conv2d

class DCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True, extra_offset_mask=True):
        super(DCN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))

        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels * 2,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.init_offset()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter):

        feat_degradation = torch.cat([input_feat, inter], dim=1)
        out = self.conv_offset_mask(feat_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(input_feat.contiguous(), offset, mask, self.weight, self.bias, self.stride,
                                       self.padding, self.dilation, self.groups, self.deformable_groups)


class SFT_layer(nn.Module):
    def __init__(self, channels_in=64, channels_out=64):
        super(SFT_layer, self).__init__()
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_out, channels_out, 1, 1, 0, bias=False),
        )

    def forward(self, x, inter):
        gamma = self.conv_gamma(inter)
        beta = self.conv_beta(inter)
        return x * gamma + beta


class HPFB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dcn = DCN_layer(c, c, 3, padding=1, bias=False)
        self.sft = SFT_layer(c, c)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, inp, p):
        dcn_out = self.dcn(inp, p)
        sft_out = self.sft(inp, p)
        out = sft_out + dcn_out +inp
        return out


class HPTM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block1 = HPFB(c)
        self.block2 = HPFB(c)
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, inp, p):
        # inp: feature map B*C*H*W
        # p: hybrid degradation-content representation B*C*H*W
        out = self.relu(self.block1(inp, p))
        out = self.relu(self.conv1(out))
        out = self.relu(self.block2(out, p))
        out = self.conv2(out) + inp
        return out


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x


class Fusion(nn.Module):
    def __init__(self, c):
        super(Fusion, self).__init__()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,groups=1, bias=True)
        )

    def forward(self, c, i):
        x = self.sca(i) * c + i
        return x


class CVPAM(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.rb = ResB(c)
        self.fusion_L = Fusion(c)
        self.fusion_R = Fusion(c)

    def forward(self, x_l, x_r, p):
        Q_l = self.l_proj1(self.rb(self.norm_l(x_l))).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.rb(self.norm_r(x_r))).permute(0, 2, 1, 3) # B, H, c, W (transposed)
        b, c, h, w = Q_l.shape
        Q_l = Q_l - torch.mean(Q_l, 3).unsqueeze(3).repeat(1, 1, 1, w)
        b2, c2, h2, w2 = Q_r_T.shape
        Q_r_T = Q_r_T - torch.mean(Q_r_T, 3).unsqueeze(3).repeat(1, 1, 1, w2)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        M = torch.softmax(attention, dim=-1)
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        F_l = self.fusion_L(F_r2l.permute(0, 3, 1, 2), x_l)
        F_r = self.fusion_R(F_l2r.permute(0, 3, 1, 2), x_r)

        return F_l, F_r, p


class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, *feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats
        new_feats = self.module(*feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.
        r1, r2, r3, r4 = new_feats
        if self.training and factor != 1.:
            r1, r2 = tuple([x+factor*(new_x-x) for x, new_x in zip(feats[:2], new_feats[:2])])
        return r1, r2, r3, r4


class HDCABlock(nn.Module):
    def __init__(self, c, fusion=True):
        super().__init__()
        self.blk = HPTM(c)
        self.fusion = CVPAM(c) if fusion else None

    def forward(self, *feats):
        new_feats = tuple([self.blk(x, feats[2]) for x in feats[:2]])
        if self.fusion:
            new_feats = self.fusion(*new_feats, feats[2])
        return new_feats


class HDCAnet(nn.Module):
    def __init__(self, up_scale=4, width=64, num_blks=16, img_channel=3, drop_path_rate=0.1, fusion_from=-1, fusion_to=1000, dual=True):
        super().__init__()
        self.dual = dual    # dual input for stereo SR (left view, right view)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.body = MySequential(
            *[DropPath(
                drop_path_rate,
                HDCABlock(
                    width,
                    fusion=(fusion_from <= i and i <= fusion_to),
                )) for i in range(num_blks)]
        )
        self.up = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=width * up_scale ** 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.PixelShuffle(up_scale),
            nn.Conv2d(in_channels= width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, bias =True)
        )
        self.up_scale = up_scale

    def forward(self, inp):
        # inp[0]: feature map B*C*H*W
        # inp[1]: hybrid degradation-content representation B*C
        p = inp[1]
        inp_hr = F.interpolate(inp[0], scale_factor=self.up_scale, mode='bilinear')
        if self.dual:
            inp = inp[0].chunk(2, dim=1)
        else:
            inp = (inp, )
        feats = [self.intro(x) for x in inp]
        feats = self.body(*feats, p, *inp)

        out = torch.cat([self.up(x) for x in feats[:2]], dim=1)
        out = out + inp_hr
        return out


@ARCH_REGISTRY.register()
class HDCA(Local_Base, HDCAnet):
    # d_size means degradation representation size
    def __init__(self, *args, train_size=(1, 6, 30, 90), d_size=(1, 64, 30, 90), fast_imp=False, fusion_from=-1, fusion_to=1000, **kwargs):
        Local_Base.__init__(self)
        HDCAnet.__init__(self, *args, img_channel=3, fusion_from=fusion_from, fusion_to=fusion_to, dual=True, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, d_size=d_size, fast_imp=fast_imp)


class Prior_Encoder(nn.Module):
    def __init__(self):
        super(Prior_Encoder, self).__init__()
        self.E_pre = ResBlock(in_feat=3, out_feat=64, stride=1)
        self.E = nn.Sequential(
            ResBlock(in_feat=64, out_feat=128, stride=2),
            ResBlock(in_feat=128, out_feat=256, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        inter = self.E_pre(x)
        fea = self.E(inter).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)

        return fea, out, inter


### Training Stage 1 ###
@ARCH_REGISTRY.register()
class BlindSSR_E(nn.Module):
    def __init__(self):
        super(BlindSSR_E, self).__init__()
        # Encoder
        self.E = MoCo(base_encoder=Prior_Encoder)

    def forward(self, x):
        x_l, x_r = x.chunk(2, dim=1)
        x_query = x_l
        x_key = torch.flip(x_r, dims=[2]) # augment
        _, z1, z2, inter = self.E(x_query, x_key)

        return _, z1, z2


### Training Stage 2 ###
@ARCH_REGISTRY.register()
class BlindSSR(nn.Module):
    def __init__(self):
        super(BlindSSR, self).__init__()
        # Generator
        self.G = HDCA()
        # Encoder
        self.E = MoCo(base_encoder=Prior_Encoder)

    def forward(self, x):
        x_l, x_r = x.chunk(2, dim=1)
        x_query = x_l
        x_key = torch.flip(x_r, dims=[2]) # augment
        _, z1, z2, inter = self.E(x_query, x_key)
        sr = self.G([x, inter])

        return sr, z1, z2


if __name__ == '__main__':
    # params = count_parameters(BlindSSR())
    # print("parameters num is %.2f M" % (params / 10 ** 6))

    ### Test Model Complexity ###
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    input = torch.randn(1, 6, 128, 128)
    degradation = torch.randn(1, 64, 128, 128)
    x = [input, degradation]
    model = BlindSSR()
    print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))






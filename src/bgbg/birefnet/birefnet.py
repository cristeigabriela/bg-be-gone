# Vendored from BiRefNet (github.com/ZhengPeng7/BiRefNet, MIT) at d8c386b,
# models/birefnet.py, cut to inference. The config the published checkpoints
# were trained under is fixed in place:
#   mul_scl_ipt = 'cat', cxt_num = 3, squeeze_block = 'BasicDecBlk_x1',
#   dec_ipt = True, dec_ipt_split = True, ms_supervision = True,
#   out_ref = True, batch_size = 4, no classifier, no refiner, no ender.
# Training-only modules (the multi-scale supervision heads and the gradient
# predictors) are still built, so the checkpoints load with strict=True, but
# eval-mode forward never runs them, exactly as in the original.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import BasicDecBlk, BasicLatBlk, SimpleConvs
from .swin_v1 import swin_v1_l, swin_v1_t

# Per backbone: the constructor, and Swin's output channels (config.py's
# lateral_channels_in_collection before the 'cat' multi-scale doubling).
BACKBONES = {
    "swin_v1_l": (swin_v1_l, [1536, 768, 384, 192]),
    "swin_v1_t": (swin_v1_t, [768, 384, 192, 96]),
}


def _up(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class BiRefNet(nn.Module):
    def __init__(self, bb="swin_v1_l"):
        super().__init__()
        build, lateral = BACKBONES[bb]
        self.bb = build()
        channels = [c * 2 for c in lateral]            # mul_scl_ipt == 'cat'
        self.cxt = channels[1:][::-1][-3:]              # cxt_num == 3
        self.squeeze_module = nn.Sequential(BasicDecBlk(channels[0] + sum(self.cxt), channels[0]))
        self.decoder = Decoder(channels)

    def forward_enc(self, x):
        x1, x2, x3, x4 = self.bb(x)
        _, _, H, W = x.shape
        x1_, x2_, x3_, x4_ = self.bb(_up(x, (H // 2, W // 2)))
        x1 = torch.cat([x1, _up(x1_, x1.shape[2:])], dim=1)
        x2 = torch.cat([x2, _up(x2_, x2.shape[2:])], dim=1)
        x3 = torch.cat([x3, _up(x3_, x3.shape[2:])], dim=1)
        x4 = torch.cat([x4, _up(x4_, x4.shape[2:])], dim=1)
        x4 = torch.cat((*[_up(x1, x4.shape[2:]), _up(x2, x4.shape[2:]),
                          _up(x3, x4.shape[2:])][-len(self.cxt):], x4), dim=1)
        return x1, x2, x3, x4

    def forward(self, x):
        x1, x2, x3, x4 = self.forward_enc(x)
        x4 = self.squeeze_module(x4)
        return self.decoder([x, x1, x2, x3, x4])


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # dec_ipt with dec_ipt_split: the input image, cut into patches the
        # size of each stage, is fed back in at every decoder stage.
        self.ipt_blk5 = SimpleConvs(2**10 * 3, channels[0] // 8, inter_channels=64)
        self.ipt_blk4 = SimpleConvs(2**8 * 3, channels[0] // 8, inter_channels=64)
        self.ipt_blk3 = SimpleConvs(2**6 * 3, channels[1] // 8, inter_channels=64)
        self.ipt_blk2 = SimpleConvs(2**4 * 3, channels[2] // 8, inter_channels=64)
        self.ipt_blk1 = SimpleConvs(2**0 * 3, channels[3] // 8, inter_channels=64)

        self.decoder_block4 = BasicDecBlk(channels[0] + channels[0] // 8, channels[1])
        self.decoder_block3 = BasicDecBlk(channels[1] + channels[0] // 8, channels[2])
        self.decoder_block2 = BasicDecBlk(channels[2] + channels[1] // 8, channels[3])
        self.decoder_block1 = BasicDecBlk(channels[3] + channels[2] // 8, channels[3] // 2)
        self.conv_out1 = nn.Sequential(nn.Conv2d(channels[3] // 2 + channels[3] // 8, 1, 1, 1, 0))

        self.lateral_block4 = BasicLatBlk(channels[1], channels[1])
        self.lateral_block3 = BasicLatBlk(channels[2], channels[2])
        self.lateral_block2 = BasicLatBlk(channels[3], channels[3])

        # Training-only heads (ms_supervision; out_ref's gradient predictors).
        self.conv_ms_spvn_4 = nn.Conv2d(channels[1], 1, 1, 1, 0)
        self.conv_ms_spvn_3 = nn.Conv2d(channels[2], 1, 1, 1, 0)
        self.conv_ms_spvn_2 = nn.Conv2d(channels[3], 1, 1, 1, 0)
        _N = 16

        def gdt_convs(c):
            return nn.Sequential(nn.Conv2d(c, _N, 3, 1, 1), nn.BatchNorm2d(_N),
                                 nn.ReLU(inplace=True))

        self.gdt_convs_4 = gdt_convs(channels[1])
        self.gdt_convs_3 = gdt_convs(channels[2])
        self.gdt_convs_2 = gdt_convs(channels[3])
        self.gdt_convs_pred_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_pred_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        # out_ref's gradient attention: used at inference too.
        self.gdt_convs_attn_4 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_3 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))
        self.gdt_convs_attn_2 = nn.Sequential(nn.Conv2d(_N, 1, 1, 1, 0))

    @staticmethod
    def get_patches_batch(x, p):
        _size_h, _size_w = p.shape[2:]
        patches_batch = []
        for idx in range(x.shape[0]):
            columns_x = torch.split(x[idx], split_size_or_sections=_size_w, dim=-1)
            patches_x = []
            for column_x in columns_x:
                rows = torch.split(column_x, split_size_or_sections=_size_h, dim=-2)
                patches_x += [patch.unsqueeze(0) for patch in rows]
            patches_batch.append(torch.cat(patches_x, dim=1))
        return torch.cat(patches_batch, dim=0)

    def _ipt(self, blk, x, p):
        return blk(_up(self.get_patches_batch(x, p), p.shape[2:]))

    def forward(self, features):
        x, x1, x2, x3, x4 = features

        x4 = torch.cat((x4, self._ipt(self.ipt_blk5, x, x4)), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p3 = _up(p4, x3.shape[2:]) + self.lateral_block4(x3)

        _p3 = torch.cat((_p3, self._ipt(self.ipt_blk4, x, _p3)), 1)
        p3 = self.decoder_block3(_p3)
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        _p2 = _up(p3, x2.shape[2:]) + self.lateral_block3(x2)

        _p2 = torch.cat((_p2, self._ipt(self.ipt_blk3, x, _p2)), 1)
        p2 = self.decoder_block2(_p2)
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        _p1 = _up(p2, x1.shape[2:]) + self.lateral_block2(x1)

        _p1 = torch.cat((_p1, self._ipt(self.ipt_blk2, x, _p1)), 1)
        _p1 = self.decoder_block1(_p1)
        _p1 = _up(_p1, x.shape[2:])
        _p1 = torch.cat((_p1, self._ipt(self.ipt_blk1, x, _p1)), 1)
        return [self.conv_out1(_p1)]

# Vendored from BiRefNet (github.com/ZhengPeng7/BiRefNet, MIT) at d8c386b:
# models/modules/{deform_conv,aspp,decoder_blocks,lateral_blocks}.py, cut to
# the blocks the published checkpoints use, with the config those checkpoints
# were trained under fixed in place:
#   dec_att = 'ASPPDeformable', dec_channels_inter = 'fixed',
#   batch_size = 4 (so every block has BatchNorm).
# Parameter names and forward passes are unchanged.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, stride=stride,
                                     padding=self.padding, bias=True)
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size, stride=stride,
                                        padding=self.padding, bias=True)
        self.regular_conv = nn.Conv2d(in_channels, out_channels=out_channels,
                                      kernel_size=kernel_size, stride=stride,
                                      padding=self.padding, bias=bias)

    def forward(self, x):
        offset = self.offset_conv(x)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        return deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight,
                             bias=self.regular_conv.bias, padding=self.padding,
                             mask=modulator, stride=self.stride)


class _ASPPModuleDeformable(nn.Module):
    def __init__(self, in_channels, planes, kernel_size, padding):
        super().__init__()
        self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.atrous_conv(x)))


class ASPPDeformable(nn.Module):
    def __init__(self, in_channels, out_channels=None, parallel_block_sizes=(1, 3, 7)):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.in_channelster = 256
        self.aspp1 = _ASPPModuleDeformable(in_channels, self.in_channelster, 1, padding=0)
        self.aspp_deforms = nn.ModuleList([
            _ASPPModuleDeformable(in_channels, self.in_channelster, conv_size,
                                  padding=int(conv_size // 2))
            for conv_size in parallel_block_sizes
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, self.in_channelster, 1, stride=1, bias=False),
            nn.BatchNorm2d(self.in_channelster),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(self.in_channelster * (2 + len(self.aspp_deforms)),
                               out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x_aspp_deforms = [aspp_deform(x) for aspp_deform in self.aspp_deforms]
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, *x_aspp_deforms, x5), dim=1)
        return self.dropout(self.relu(self.bn1(self.conv1(x))))


class BasicDecBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        inter_channels = 64
        self.conv_in = nn.Conv2d(in_channels, inter_channels, 3, 1, padding=1)
        self.relu_in = nn.ReLU(inplace=True)
        self.dec_att = ASPPDeformable(in_channels=inter_channels)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, padding=1)
        self.bn_in = nn.BatchNorm2d(inter_channels)
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu_in(self.bn_in(self.conv_in(x)))
        x = self.dec_att(x)
        return self.bn_out(self.conv_out(x))


class BasicLatBlk(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)


class SimpleConvs(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.conv_out = nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        return self.conv_out(self.conv1(x))

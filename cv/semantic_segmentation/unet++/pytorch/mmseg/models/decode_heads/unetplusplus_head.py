# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..builder import HEADS
from .decode_head import BaseDecodeHead

def conv_bn_relu(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   stride=stride),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))

# class AttBlock(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super(AttBlock, self).__init__()
#         self.W_g = nn.Sequential(
#             nn.Conv2d(in_channels=F_g,
#                       out_channels=F_int,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(in_channels=F_l,
#                       out_channels=F_int,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0,
#                       bias=True),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(in_channels=F_int,
#                       out_channels=1,
#                       kernel_size=1,
#                       stride=1,
#                       padding=0,
#                       bias=True),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         print(g.shape, x.shape)
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         print(g1.shape, x1.shape)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)

#         return x * psi


class DecBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels
    ):
        super().__init__()
        self.conv1 = conv_bn_relu(in_channels=in_channels + skip_channels,
                                  out_channels=out_channels)

        self.conv2 = conv_bn_relu(in_channels=out_channels,
                                  out_channels=out_channels)

        self.up = nn.Upsample(scale_factor=2,
                              mode='bilinear',
                              align_corners=True)

        # self.att = AttBlock(F_g=in_channels, F_l=skip_channels, F_int=in_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        return x


@HEADS.register_module()
class UNetPlusPlusHead(BaseDecodeHead):
    def __init__(
            self, **kwargs
    ):
        super(UNetPlusPlusHead, self).__init__(**kwargs)

        self.decoder_layers = nn.ModuleList()
        skip_channels = self.in_channels[:-1]

        blocks = {}
        for stage_idx in range(1, len(self.in_channels)):
            for lvl_idx in range(len(self.in_channels) - stage_idx):
                in_ch = self.in_channels[lvl_idx + 1]
                skip_ch = skip_channels[lvl_idx] * (stage_idx)
                out_ch = self.in_channels[lvl_idx]
                blocks[f'x_{lvl_idx}_{stage_idx}'] = DecBlock(in_ch, skip_ch, out_ch)

        self.blocks = nn.ModuleDict(blocks)

    def forward(self, features):
        dense_x = OrderedDict()
        for idx, item in enumerate(features):
            dense_x[f'x_{idx}_{0}'] = features[idx]

        for stage_idx in range(1, len(self.in_channels)):
            for lvl_idx in range(len(self.in_channels) - stage_idx):
                skip_features = [dense_x[f'x_{lvl_idx}_{idx}'] for idx in range(stage_idx)]
                skip_features = torch.cat(skip_features, dim=1)
                output = self.blocks[f'x_{lvl_idx}_{stage_idx}'](dense_x[f'x_{lvl_idx + 1}_{stage_idx - 1}'],
                                                                 skip_features)
                dense_x[f'x_{lvl_idx}_{stage_idx}'] = output


        res = self.cls_seg(dense_x[next(reversed(dense_x))])
        return res
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

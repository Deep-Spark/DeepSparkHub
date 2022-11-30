# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Copyright (c) TorchSeg. All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

# encoding: utf-8
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import resnet101
from ..seg_oprs import ConvBnRelu


def psanet_resnet101v1c(num_classes):
    return PSPNet(num_classes, None)


class PSPNet(nn.Module):
    def __init__(self, out_planes, criterion, pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        super(PSPNet, self).__init__()
        self.backbone = resnet101(norm_layer=norm_layer,
                                  bn_eps=1e-5,
                                  bn_momentum=0.1,
                                  deep_stem=True, stem_width=64)
        self.backbone.layer3.apply(partial(self._nostride_dilate, dilate=2))
        self.backbone.layer4.apply(partial(self._nostride_dilate, dilate=4))

        self.business_layer = []
        self.psa_layer = PointwiseSpatialAttention('psa', out_planes, 2048,
                                                   norm_layer=norm_layer)
        self.aux_layer = nn.Sequential(
            ConvBnRelu(1024, 1024, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(1024, out_planes, kernel_size=1)
        )
        self.business_layer.append(self.psa_layer)
        self.business_layer.append(self.aux_layer)

        self.criterion = criterion

    def forward(self, data, label=None):
        blocks = self.backbone(data)

        psa_fm = self.psa_layer(blocks[-1])
        aux_fm = self.aux_layer(blocks[-2])

        psa_fm = F.interpolate(psa_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)
        aux_fm = F.interpolate(aux_fm, scale_factor=8, mode='bilinear',
                               align_corners=True)

        if self.training:
            return psa_fm, aux_fm
        else:
            return psa_fm

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


class PointwiseSpatialAttention(nn.Module):
    def __init__(self, name, out_planes, fc_dim=4096, pool_scales=[1, 2, 3, 6],
                 norm_layer=nn.BatchNorm2d):
        super(PointwiseSpatialAttention, self).__init__()

        self.inner_channel = 512
        self.collect_reduction = ConvBnRelu(fc_dim, 512, 1, 1, 0,
                                            has_bn=True, has_relu=True,
                                            has_bias=False,
                                            norm_layer=norm_layer)
        self.collect_attention = nn.Sequential(
            ConvBnRelu(512, 512, 1, 1, 0,
                       has_bn=True, has_relu=True,
                       has_bias=False, norm_layer=norm_layer),
            ConvBnRelu(512, 3600, 1, 1, 0,
                       has_bn=False, has_relu=False,
                       has_bias=False, norm_layer=norm_layer)
        )

        self.distribute_reduction = ConvBnRelu(fc_dim, 512, 1, 1, 0,
                                               has_bn=True, has_relu=True,
                                               has_bias=False,
                                               norm_layer=norm_layer)
        self.distribute_attention = nn.Sequential(
            ConvBnRelu(512, 512, 1, 1, 0,
                       has_bn=True, has_relu=True,
                       has_bias=False, norm_layer=norm_layer),
            ConvBnRelu(512, 3600, 1, 1, 0,
                       has_bn=False, has_relu=False,
                       has_bias=False, norm_layer=norm_layer)
        )

        self.proj = ConvBnRelu(1024, 2048, 1, 1, 0,
                               has_bn=True, has_relu=True,
                               has_bias=False, norm_layer=norm_layer)

        self.conv6 = nn.Sequential(
            ConvBnRelu(fc_dim + len(pool_scales) * 512, 512, 3, 1, 1,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer),
            nn.Dropout2d(0.1, inplace=False),
            nn.Conv2d(512, out_planes, kernel_size=1)
        )

    def forward(self, x):
        collect_reduce_x = self.collect_reduction(x)
        collect_attention = self.collect_attention(collect_reduce_x)
        b, c, h, w = collect_attention.size()
        collect_attention = collect_attention.view(b, c, -1)
        collect_reduce_x = collect_reduce_x.view(b, self.inner_channel, -1)
        collect_fm = torch.bmm(collect_reduce_x,
                               torch.softmax(collect_attention, dim=1))
        collect_fm = collect_fm.view(b, self.inner_channel, h, w)

        distribute_reduce_x = self.distribute_reduction(x)
        distribute_attention = self.distribute_attention(distribute_reduce_x)
        b, c, h, w = distribute_attention.size()
        distribute_attention = distribute_attention.view(b, c, -1)
        distribute_reduce_x = distribute_reduce_x.view(b, self.inner_channel,
                                                       -1)
        distribute_fm = torch.bmm(distribute_reduce_x,
                                  torch.softmax(distribute_attention, dim=1))
        distribute_fm = distribute_fm.view(b, self.inner_channel, h, w)

        psa_fm = torch.cat([collect_fm, distribute_fm], dim=1)
        psa_fm = self.proj(psa_fm)
        fm = torch.cat([x, psa_fm], dim=1)

        out = self.conv6(fm)
        return out

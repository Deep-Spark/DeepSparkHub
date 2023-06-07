# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
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


import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .decode_head import BaseDecodeHead

BatchNorm2d = nn.SyncBatchNorm
bn_mom = 0.1


class DDRModule(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 align_corners,
                 scale_factor=None):
        super(DDRModule, self).__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.num_classes = num_classes
        self.scale_factor = scale_factor
        self.align_corners = align_corners

        self.bn1 = BatchNorm2d(self.in_channels, momentum=bn_mom)
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(self.channels, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.channels, self.num_classes, kernel_size=1, padding=0, bias=True)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward function."""


        y = self.conv1(self.relu(self.bn1(x)))
        output = self.conv2(self.relu(self.bn2(y)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            output = F.interpolate(output, size=[height, width], mode='bilinear')

        return output
    
@HEADS.register_module()
class DDRHead(BaseDecodeHead):
    def __init__(self, scale_factor=None, **kwargs):
        super(DDRHead, self).__init__(**kwargs)
        self.ddr_module = DDRModule(
            self.in_channels,
            self.channels,
            self.num_classes,
            self.align_corners,
            scale_factor
        )
    
    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        output = self.ddr_module(x)
        # output = self.cls_seg(output)
        return output
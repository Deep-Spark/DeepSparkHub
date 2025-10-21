# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import os
import sys

import torchvision

sys.path.append("../../torchvision/pytorch")

from train import train_model
from utils import padding_conv_channel_to_4
import dpn

def create_model(args):
    model = dpn.__dict__[args.model](pretrained=args.pretrained, num_classes=args.num_classes)
    if args.nhwc:
        args.padding_channel = True
        model.features[0].conv = padding_conv_channel_to_4(model.features[0].conv)
    return model


train_model(create_model)

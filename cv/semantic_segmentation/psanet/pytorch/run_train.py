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

sys.path.append("../../torchvision/pytorch")

from model import psanet as builder
from train import train_model
from utils import padding_conv_channel_to_4


def create_model(args):
    model = builder.__dict__[args.model](args.num_classes)
    if args.padding_channel:
        model.backbone.conv1[0] = padding_conv_channel_to_4(model.backbone.conv1[0])

    return model


train_model(create_model)



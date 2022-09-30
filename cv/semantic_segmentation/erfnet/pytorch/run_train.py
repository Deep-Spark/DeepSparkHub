# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.

import os
import sys

sys.path.append("../../torchvision/pytorch")

import model as erfnet
from train import train_model
from utils import padding_conv_channel_to_4


def create_model(args):
    args.find_unused_parameters = True
    model = erfnet.__dict__[args.model](args.num_classes)
    if args.padding_channel:
        model.encoder.initial_block.conv = padding_conv_channel_to_4(model.encoder.initial_block.conv)

    return model


train_model(create_model)



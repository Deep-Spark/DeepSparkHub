# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
import os
import sys

import torchvision

sys.path.append("../../torchvision/pytorch")

from train import train_model
from utils import padding_conv_channel_to_4

def create_model(args):
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=args.num_classes)
    if args.nhwc:
        args.padding_channel = True
        model.conv1 = padding_conv_channel_to_4(model.conv1)

    return model


train_model(create_model)

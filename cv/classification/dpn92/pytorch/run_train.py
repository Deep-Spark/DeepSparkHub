# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
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

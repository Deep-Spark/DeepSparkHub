# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
import os
import sys

import torchvision

sys.path.append("../../torchvision/pytorch")

from train import train_model
from utils import padding_conv_channel_to_4
import resnest

def create_model(args):
    model = resnest.__dict__[args.model](pretrained=args.pretrained, num_classes=args.num_classes)
    args.padding_channel = False

    return model


train_model(create_model)

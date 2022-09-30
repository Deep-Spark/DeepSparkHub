# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
import os
import sys

import torchvision

sys.path.append("../../torchvision/pytorch")

from train import train_model
from utils import padding_conv_channel_to_4

def create_model(args):
    # How does it work?
    # - Apex.amp couldn't solve namedtuple input, thus produces the problem:
    #   TypeError: __new__() missing 1 required positional argument: 'aux_logits'
    # - See https://github.com/pytorch/vision/issues/1048 for discussions
    torchvision.models.inception.InceptionOutputs = lambda a,b:(a,b)

    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, num_classes=args.num_classes)
    args.padding_channel = False

    return model


train_model(create_model)

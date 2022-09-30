# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.

import os
import sys

sys.path.append("../../torchvision/pytorch")

from model import create_segnet
from train import train_model
from utils import padding_conv_channel_to_4


def create_model(args):
    model = create_segnet(args.model, num_classes=args.num_classes)
    if args.padding_channel:
        if args.model == "segnet_resnet":
            model.stage1[0] = padding_conv_channel_to_4(model.stage1[0])
        else:
            model.conv11 = padding_conv_channel_to_4(model.conv11)

    return model


train_model(create_model)



import copy
import math
from torch.utils import checkpoint

from ssd300 import SSD300


def convert_model(model, config):
    model_options = {
        'use_nhwc': config.nhwc,
        'pad_input': config.pad_input,
        'bn_group': config.bn_group,
    }
    model = SSD300(config, config.num_classes, **model_options).cuda()
    return model

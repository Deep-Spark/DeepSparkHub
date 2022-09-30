import os
import torch
import logging
import torch.utils.model_zoo as model_zoo

from ...utils.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")
BACKBONE_REGISTRY.__doc__ = """
Registry for backbone, i.e. resnet.

The registered object will be called with `obj()`
and expected to return a `nn.Module` object.
"""

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnet50c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet50-25c4b509.pth',
    'resnet101c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet101-2a57e44d.pth',
    'resnet152c': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/resnet152-0d43d698.pth',
    'xception65': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/tf-xception65-270e81cf.pth',
    'hrnet_w18_small_v1': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/hrnet-w18-small-v1-08f8ae64.pth',
    'mobilenet_v2': 'https://github.com/LikeLy-Journey/SegmenTron/releases/download/v0.1.0/mobilenetV2-15498621.pth',
}


def get_segmentation_backbone(backbone, cfg=None, norm_layer=torch.nn.BatchNorm2d):
    """
    Built the backbone model, defined by `cfg.MODEL.BACKBONE`.
    """
    model = BACKBONE_REGISTRY.get(backbone)(cfg=cfg, norm_layer=norm_layer)
    return model

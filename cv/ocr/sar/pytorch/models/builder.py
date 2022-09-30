# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from sar.cnn import ACTIVATION_LAYERS as MMCV_ACTIVATION_LAYERS
from sar.cnn import UPSAMPLE_LAYERS as MMCV_UPSAMPLE_LAYERS
from ocrcv.utils import Registry, build_from_cfg
from ocrdet.models.builder import BACKBONES as MMDET_BACKBONES

RECOGNIZERS = Registry('recognizer')
CONVERTORS = Registry('convertor')
ENCODERS = Registry('encoder')
DECODERS = Registry('decoder')
PREPROCESSOR = Registry('preprocessor')

UPSAMPLE_LAYERS = Registry('upsample layer', parent=MMCV_UPSAMPLE_LAYERS)
BACKBONES = Registry('models', parent=MMDET_BACKBONES)
LOSSES = BACKBONES
DETECTORS = BACKBONES
ROI_EXTRACTORS = BACKBONES
HEADS = BACKBONES
NECKS = BACKBONES

ACTIVATION_LAYERS = Registry('activation layer', parent=MMCV_ACTIVATION_LAYERS)


def build_recognizer(cfg, train_cfg=None, test_cfg=None):
    """Build recognizer."""
    return build_from_cfg(cfg, RECOGNIZERS,
                          dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_convertor(cfg):
    """Build label convertor for scene text recognizer."""
    return build_from_cfg(cfg, CONVERTORS)


def build_encoder(cfg):
    """Build encoder for scene text recognizer."""
    return build_from_cfg(cfg, ENCODERS)


def build_decoder(cfg):
    """Build decoder for scene text recognizer."""
    return build_from_cfg(cfg, DECODERS)


def build_preprocessor(cfg):
    """Build preprocessor for scene text recognizer."""
    return build_from_cfg(cfg, PREPROCESSOR)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))

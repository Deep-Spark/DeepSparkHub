# Copyright (c) OpenMMLab. All rights reserved.
from models.builder import DETECTORS
from .encode_decode_recognizer import EncodeDecodeRecognizer


@DETECTORS.register_module()
class SARNet(EncodeDecodeRecognizer):
    """Implementation of `SAR <https://arxiv.org/abs/1811.00751>`_"""

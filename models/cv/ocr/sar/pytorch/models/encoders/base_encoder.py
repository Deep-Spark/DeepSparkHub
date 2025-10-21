# Copyright (c) OpenMMLab. All rights reserved.
from ocrcv.runner import BaseModule

from models.builder import ENCODERS


@ENCODERS.register_module()
class BaseEncoder(BaseModule):
    """Base Encoder class for text recognition."""

    def forward(self, feat, **kwargs):
        return feat

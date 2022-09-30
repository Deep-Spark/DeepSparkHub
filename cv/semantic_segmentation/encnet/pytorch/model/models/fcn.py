from __future__ import division

import torch.nn.functional as F

from .segbase import SegBaseModel
from ..modules import _FCNHead

__all__ = ['FCN']


class FCN(SegBaseModel):
    def __init__(self, cfg):
        super(FCN, self).__init__(cfg)
        self.head = _FCNHead(2048, self.nclass)
        if self.aux:
            self.auxlayer = _FCNHead(1024, self.nclass)

        self.__setattr__('decoder', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        size = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

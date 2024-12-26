# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# from .nms_wrapper import nms, soft_nms
from torchvision.ops import nms
soft_nms = nms 

__all__ = ['nms', 'soft_nms']

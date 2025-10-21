# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# from .roi_align import roi_align, RoIAlign
from torchvision.ops import roi_align 
from .roi_align import RoIAlign

__all__ = ['roi_align', 'RoIAlign']

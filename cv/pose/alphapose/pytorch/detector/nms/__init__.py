# from .nms_wrapper import nms, soft_nms
from torchvision.ops import nms
soft_nms = nms 

__all__ = ['nms', 'soft_nms']

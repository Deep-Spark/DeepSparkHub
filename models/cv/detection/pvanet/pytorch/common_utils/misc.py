import os
import sys
import errno
from typing import Any
from PIL import Image

import torch


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def _is_pil_image(img: Any) -> bool:
    try:
        import accimage
    except ImportError:
        accimage = None
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def get_image_size(img):
    if isinstance(img, torch.Tensor):
        return [img.shape[-1], img.shape[-2]]

    if _is_pil_image(img):
        return img.size
    raise TypeError("Unexpected type {}".format(type(img)))
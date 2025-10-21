import torch

try:
    from ixops import nms, roi_align, roi_pool
except:
    from torchvision.ops import nms, roi_align, roi_pool

__all__ = ['roi_align', 'roi_pool', 'nms', '_assert_has_ops']


def _has_ops():
    return True


def _assert_has_ops():
    if not _has_ops():
        raise RuntimeError(
            "Couldn't load custom C++ ops. This can happen if your PyTorch and "
            "torchvision versions are incompatible, or if you had errors while compiling "
            "torchvision from source. For further information on the compatible versions, check "
            "https://github.com/pytorch/vision#installation for the compatibility matrix. "
            "Please check your PyTorch version with torch.__version__ and your torchvision "
            "version with torchvision.__version__ and verify if they are compatible, and if not "
            "please reinstall torchvision so that it matches your PyTorch install."
        )

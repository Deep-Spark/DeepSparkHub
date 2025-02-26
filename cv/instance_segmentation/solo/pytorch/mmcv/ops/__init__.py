# Copyright (c) OpenMMLab. All rights reserved.
from .deprecated_wrappers import Conv2d_deprecated as Conv2d
from .deprecated_wrappers import ConvTranspose2d_deprecated as ConvTranspose2d
from .deprecated_wrappers import Linear_deprecated as Linear
from .deprecated_wrappers import MaxPool2d_deprecated as MaxPool2d
from .focal_loss import (SigmoidFocalLoss, SoftmaxFocalLoss,
                         sigmoid_focal_loss, softmax_focal_loss)
from .info import (get_compiler_version, get_compiling_cuda_version,
                   get_onnxruntime_op_path)
from .nms import batched_nms, nms, nms_match, soft_nms
from .roi_align import RoIAlign, roi_align
from .roi_pool import RoIPool, roi_pool
from .sync_bn import SyncBatchNorm
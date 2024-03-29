# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

from .train import init_random_seed, train_detector
from .utils import (disable_text_recog_aug_test, replace_image_to_tensor,
                    tensor2grayimgs)

__all__ = [
    'train_detector', 'init_random_seed',
    'replace_image_to_tensor', 'disable_text_recog_aug_test',
    'tensor2grayimgs'
]

# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

from .test import multi_gpu_test, single_gpu_test
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_detector)

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_detector', 
    'multi_gpu_test', 'single_gpu_test', 'init_random_seed'
]

# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
from .test import multi_gpu_test, single_gpu_test
from .train import set_random_seed, train_model

__all__ = [
    'train_model', 'set_random_seed', 'init_model', 
    'multi_gpu_test', 'single_gpu_test'
]

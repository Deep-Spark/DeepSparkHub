# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.

_base_ = [
    '../_base_/models/ddrnet_23_slim.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
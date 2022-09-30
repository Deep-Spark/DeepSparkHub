# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from .native_pipeline import build_native_pipeline
from .input_iterators import ConvertDaliInputIterator


"""
Build a train pipe for training (without touching the data)

returns train_pipe
"""
def prebuild_pipeline(config):
    if config.dali:
        from .dali_pipeline import prebuild_dali_pipeline
        return prebuild_dali_pipeline(config)
    else:
        return None

"""
Build a data pipeline for either training or eval

Training : returns loader, epoch_size
Eval : returns loader, inv_class_map, cocoGt
"""
def build_pipeline(config, training=True, pipe=None):
    # Handle training / testing differently due to different
    # outputs. But still want to do this to abstract out the
    # use of EncodingInputIterator and RateMatcher
    if training:
        if config.dali:
            from .dali_pipeline import build_dali_pipeline
            train_loader, epoch_size = build_dali_pipeline(config, training=True, pipe=pipe)
            train_sampler = None
            train_loader = ConvertDaliInputIterator(train_loader)
        else:
            train_loader, epoch_size, train_sampler = build_native_pipeline(config, training=True, pipe=pipe)
        return train_loader, epoch_size, train_sampler
    else:
        return build_native_pipeline(config, training=False)


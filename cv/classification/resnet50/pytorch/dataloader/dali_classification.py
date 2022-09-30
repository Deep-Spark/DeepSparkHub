# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.
# From: https://github.com/NVIDIA/DALI/blob/release_v1.6/docs/examples/use_cases/pytorch/resnet50/main.py
import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

zeros = np.zeros((1, 224, 224), dtype=np.float)
@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, nhwc=False):
    images, labels = fn.readers.file(file_root=data_dir,
                                 shard_id=shard_id,
                                 num_shards=num_shards,
                                 random_shuffle=is_training,
                                 pad_last_batch=True,
                                 name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    if nhwc:
        images = fn.cat(images, zeros, axis=0)
    labels = labels.gpu()
    return images, labels


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, size, local_rank=-1, world_size=1, nhwc=False):
    # This line fixes the bug of shard_id in fn.readers.file, if it's -1, bug happens
    local_rank = 0 if local_rank == -1 else local_rank
    if type == "train":
        pipe = create_dali_pipeline(batch_size=batch_size,
                                    num_threads=num_threads,
                                    device_id=device_id,
                                    data_dir=image_dir,
                                    crop=size,
                                    size=size,
                                    dali_cpu=True,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    is_training=True,
                                    nhwc=nhwc)
        pipe.build()
        return DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    elif type == "val":
        pipe = create_dali_pipeline(batch_size=batch_size,
                                    num_threads=num_threads,
                                    device_id=device_id,
                                    seed=12 + local_rank,
                                    data_dir=image_dir,
                                    crop=size,
                                    size=size,
                                    dali_cpu=True,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    is_training=False,
                                    nhwc=nhwc)
        pipe.build()
        return DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.


import os
import time

import torch
import torchvision
from .utils import presets_classification as presets

"""
Examples:

>>> dataset_train, dataset_val = load_data(train_dir, val_dir, args)
"""


def get_datasets(traindir,
                 valdir,
                 resize_size=256,
                 crop_size=224,
                 auto_augment_policy=None,
                 random_erase_prob=0.):
    # Data loading code
    print("Loading data")
    print("Loading training data")
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                          random_erase_prob=random_erase_prob))

    print("Loading validation data")
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size))

    return dataset, dataset_test


def get_input_size(model):
    biger_input_size_models = ['inception']
    resize_size = 256
    crop_size = 224
    for bi_model in biger_input_size_models:
        if bi_model in model:
            resize_size = 342
            crop_size = 299

    return resize_size, crop_size


def load_data(train_dir, val_dir, args):
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    resize_size, crop_size = get_input_size(args.model)
    dataset, dataset_test = get_datasets(train_dir, val_dir,
                                         auto_augment_policy=auto_augment_policy,
                                         random_erase_prob=random_erase_prob,
                                         resize_size=resize_size,
                                         crop_size=crop_size)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def _create_torch_dataloader(train_dir, val_dir, args):
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    return data_loader, data_loader_test


def _create_dali_dataloader(train_dir, val_dir, args):
    from .dali_classification import get_imagenet_iter_dali
    device = torch.cuda.current_device()
    _, crop_size = get_input_size(args.model)
    data_loader = get_imagenet_iter_dali('train', train_dir, args.batch_size,
                                         num_threads=args.workers,
                                         device_id=device,
                                         size=crop_size)
    data_loader_test = get_imagenet_iter_dali('val', train_dir, args.batch_size,
                                              num_threads=args.workers,
                                              device_id=device,
                                              size=crop_size)

    return data_loader, data_loader_test


def create_dataloader(train_dir, val_dir, args):
    print("Creating data loaders")
    if args.dali:
        train_dir = os.path.dirname(train_dir)
        val_dir = os.path.dirname(val_dir)
        return _create_dali_dataloader(train_dir, val_dir, args)
    return _create_torch_dataloader(train_dir, val_dir, args)

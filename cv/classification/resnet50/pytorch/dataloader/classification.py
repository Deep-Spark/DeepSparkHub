# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.
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
                 random_erase_prob=0.,
                 nhwc=False):
    # Data loading code
    print("Loading data")
    print("Loading training data")
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        presets.ClassificationPresetTrain(crop_size=crop_size, auto_augment_policy=auto_augment_policy,
                                          random_erase_prob=random_erase_prob, nhwc=nhwc))

    print("Loading validation data")
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        presets.ClassificationPresetEval(crop_size=crop_size, resize_size=resize_size, nhwc=nhwc))

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
                                         crop_size=crop_size,
                                         nhwc=args.channels_last)
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
                                         size=crop_size,
                                         local_rank=args.local_rank, world_size=args.world_size,
                                         nhwc=args.channels_last
                                         )
    data_loader_test = get_imagenet_iter_dali('val', val_dir, args.batch_size,
                                              num_threads=args.workers,
                                              device_id=device,
                                              local_rank=args.local_rank, world_size=args.world_size,
                                              nhwc=args.channels_last,
                                              size=crop_size)

    return data_loader, data_loader_test


def create_dataloader(train_dir, val_dir, args):
    print("Creating data loaders")
    if args.dali:
        return _create_dali_dataloader(train_dir, val_dir, args)
    return _create_torch_dataloader(train_dir, val_dir, args)

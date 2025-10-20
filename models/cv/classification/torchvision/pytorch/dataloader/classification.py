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
import numpy as np

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
    resize_size, crop_size = args.crop_size, args.base_size
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
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator
    from nvidia.dali.plugin.pytorch import LastBatchPolicy
    from . import dali_classification

    if "LOCAL_RANK" in os.environ:
        device_id = int(os.environ["LOCAL_RANK"])
    else:
        device_id = args.local_rank

    if device_id < 0:
        device_id = 0

    crop_size = args.crop_size
    dali_classification.zeros = np.zeros((1, crop_size, crop_size), dtype="float32")
    data_loader = dali_classification.create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=device_id,
                                seed=12 + device_id,
                                data_dir=train_dir,
                                crop=crop_size,
                                size=crop_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=device_id,
                                num_shards=args.world_size,
                                is_training=True,
                                padding_channel=args.padding_channel)
    data_loader_test = dali_classification.create_dali_pipeline(batch_size=args.batch_size,
                                num_threads=args.workers,
                                device_id=device_id,
                                seed=12 + device_id,
                                data_dir=val_dir,
                                crop=crop_size,
                                size=crop_size,
                                dali_cpu=args.dali_cpu,
                                shard_id=device_id,
                                num_shards=args.world_size,
                                is_training=False,
                                padding_channel=args.padding_channel)

    data_loader.build()
    data_loader = PyTorchIterator(data_loader,  reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)

    data_loader_test.build()
    data_loader_test = PyTorchIterator(data_loader_test,  reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return data_loader, data_loader_test


def create_dataloader(train_dir, val_dir, args):
    print("Creating data loaders")
    if args.dali:
        # train_dir = os.path.dirname(train_dir)
        # val_dir = os.path.dirname(val_dir)
        return _create_dali_dataloader(train_dir, val_dir, args)
    return _create_torch_dataloader(train_dir, val_dir, args)

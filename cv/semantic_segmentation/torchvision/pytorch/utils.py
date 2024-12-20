# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from collections import defaultdict, deque
import datetime
import time
import torch
import torch.distributed as dist

import errno
import os

from common_utils import *

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def nhwc_collate_fn(fp16=False, padding_channel=False):
    dtype = torch.float32
    if fp16:
        dtype = torch.float16
    def _collect_fn(batch):
        batch = collate_fn(batch)
        if not padding_channel:
            return batch
        batch = list(batch)
        image = batch[0]
        zeros = image.new_zeros(image.shape[0], image.shape[2], image.shape[3], 1)
        image = torch.cat([image.permute(0, 2, 3, 1), zeros], dim=-1).permute(0, 3, 1, 2)
        image = image.to(memory_format=torch.channels_last, dtype=dtype)
        batch[0] = image
        return batch

    return _collect_fn


def padding_conv_channel_to_4(conv: torch.nn.Conv2d):
    new_conv = torch.nn.Conv2d(
        4, conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        bias=conv.bias is not None
    )
    weight_shape = conv.weight.shape
    padding_weight = conv.weight.new_zeros(weight_shape[0], 1, *weight_shape[2:])
    new_conv.weight = torch.nn.Parameter(torch.cat([conv.weight, padding_weight], dim=1))
    new_conv.bias = conv.bias
    return new_conv

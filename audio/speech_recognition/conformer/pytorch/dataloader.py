# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.

import numpy as np
import random

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


def get_sampler(dataset, sampler_type):
    return dict(
        random=RandomSampler,
        sequential=SequentialSampler,
        distributed=DistributedSampler
    )[sampler_type.lower()](dataset)


class WorkerInitializer(object):

    _instance = None

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, idx):
        np.random.seed(seed=self.seed + idx)
        random.seed(self.seed + idx)

    @classmethod
    def default(cls, seed=0):
        if cls._instance is None:
            cls._instance = cls(seed)
        return cls._instance


# sampler: Random | Sequential | Distributed
def create_dataloader(
        dataset,
        batch_size,
        worker_init_fn: WorkerInitializer=None,
        sampler_type='Random',
        pin_memory=True
):
    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default()
    sampler = get_sampler(dataset, sampler_type)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=0 if batch_size <= 8 else 4,
        worker_init_fn=worker_init_fn,
        pin_memory=pin_memory,
        collate_fn=padding_collate_fn
    )

    return dataloader


def padding_collate_fn(batch, pad_id: int = 0):
    r"""
    Functions that pad to the maximum sequence length

    Args:
        batch (tuple): tuple contains input and target tensors
        pad_id (int): identification of pad token

    Returns:
        seqs (torch.FloatTensor): tensor contains input sequences.
        target (torch.IntTensor): tensor contains target sequences.
        seq_lengths (torch.IntTensor): tensor contains input sequence lengths
        target_lengths (torch.IntTensor): tensor contains target sequence lengths
    """
    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) - 1 for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)

    return seqs, targets, seq_lengths, target_lengths

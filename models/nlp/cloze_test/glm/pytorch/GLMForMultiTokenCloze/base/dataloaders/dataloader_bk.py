# coding=utf-8
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.dataloader import default_collate
import h5py
import glob
import os

from utils import print_rank_0


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


class H5pyDataSet(Dataset):
    def __init__(self, split, args):
        self.split = split
        self.args = args
        self.load()

    def load(self):
        # used h5py load data
        if self.split == 'train':
            files = glob.glob(os.path.join(self.args.train_data_dir, "*.hdf5"))
        elif self.split == 'eval':
            files = glob.glob(os.path.join(self.args.eval_data_dir, "*.hdf5"))
        else:
            assert "split should be 'train' or 'evel'"
        files = sorted(files)
        self.data_files = []
        self.data_length = 0
        self.bins = []
        for path in files:
            with h5py.File(path, 'r') as f:
                self.data_files.append(path)
                self.bins.append(
                    [self.data_length, self.data_length+f['text'].shape[0]])
                self.data_length += f['text'].shape[0]

    def __len__(self):
        return self.data_length

    def _get_index(self, idx):
        for i, (start_idx, end_idx) in enumerate(self.bins):
            if idx >= start_idx and idx < end_idx:
                num_bin = i
                num_data = idx - start_idx
                return num_bin, num_data

    def __getitem__(self, idx):
        num_bin, num_data = self._get_index(idx)
        with h5py.File(self.data_files[num_bin], 'r') as f:
            keys = f.keys()
            sample = {}
            for key in keys:
                sample[key] = f[key][num_data]
            sample['uid'] = idx
        return sample


def my_collate(batch):
    text_list = []
    new_batch = []
    for sample in batch:
        new_sample = {key: value for key,
                      value in sample.items() if key != 'uid'}
        new_sample['label'] = 0
        text_list.append(sample['text'])
        new_batch.append(new_sample)

    def pad_choice_dim(data, choice_num):
        if len(data) < choice_num:
            data = np.concatenate([data] + [data[0:1]]
                                  * (choice_num - len(data)))
        return data

    if len(text_list[0].shape) == 2:
        choice_nums = list(map(len, text_list))
        max_choice_num = max(choice_nums)
        for i, sample in enumerate(new_batch):
            for key, value in sample.items():
                if key not in ['answer_idx', 'label']:
                    sample[key] = pad_choice_dim(value, max_choice_num)
                else:
                    sample[key] = value

    new_batch = default_collate(new_batch)
    if 'uid' in batch[0]:
        uid_list = [sample['uid'] for sample in batch]
        new_batch['uid'] = uid_list

    return new_batch


def build_data_loader(dataset, batch_size, num_workers, drop_last, shuffle=True, only_rank0=False, worker_init_fn: WorkerInitializer = None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""
    if worker_init_fn is None:
        worker_init_fn = WorkerInitializer.default()
    world_size = dist.get_world_size()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, shuffle=shuffle)
    print_rank_0(
        f"use sampler: DistributedSampler, num_replicas:{world_size}")

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              drop_last=drop_last,
                                              pin_memory=True,
                                              collate_fn=my_collate,
                                              worker_init_fn=worker_init_fn)
    return data_loader


def build_train_dataloader(args, worker_init_fn: WorkerInitializer = None):
    """Traing and validation dataloaders."""
    print_rank_0('building train dataloaders ...')
    train_dataset = H5pyDataSet("train", args)
    train_dataloader = build_data_loader(
        train_dataset, args.train_batch_size, args.num_workers, drop_last=False, worker_init_fn=worker_init_fn)

    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    print_rank_0(
        f'train samples:{len(train_dataset)}, batch size:{args.train_batch_size}, epoch:{args.epochs}, train_iters_per_epoch:{args.train_iters_per_epoch}')
    return train_dataloader

def build_eval_dataloaders(args):
    print_rank_0('building eval dataloaders ...')
    eval_dataset = H5pyDataSet("eval", args)
    eval_dataloader = build_data_loader(
        eval_dataset, args.eval_batch_size, args.num_workers, shuffle=False, drop_last=False)
    print_rank_0(
        f'eval samples:{len(eval_dataset)}, batch size:{args.eval_batch_size}')
    return eval_dataloader

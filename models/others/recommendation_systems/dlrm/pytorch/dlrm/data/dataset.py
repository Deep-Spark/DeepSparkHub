import concurrent
import functools
import math
import os
import queue

import numpy as np
from dlrm import mlperf_logger

import torch
from torch.utils.data import Dataset

from dlrm.utils import distributed as dist


def get_data_loader(dataset_path,
                    batch_size,
                    test_batch_size,
                    return_device="cuda",
                    dataset_type="bin",
                    num_workers=0,
                    shuffle=False,
                    **kwargs):
    """Create data loaders

    Args:
        dataset_path (str): Path to dataset, train_data.bin and test_data.bin must exist under dataset_path.
        batch_size (int): Training batch size
        test_batch_size (int): Test batch size
        return_device (str): Where to put the returned data. Default 'cuda'
        dataset_type (str): One of ["bin", "memmap", "dist"] indicates which dataset to use. Default "bin".
        num_works (int): Default 0
        shuffle (bool): If True, shuffle batch order. Default False.

    Keyword Arguments:
        numerical_features(boolean): If True, load numerical features for bottom_mlp. Default False
        categorical_features (list or None): categorical features used by the rank

    Returns:
        data_loader_train (DataLoader):
        data_loader_test (DataLoader):
    """
    train_dataset_bin = os.path.join(dataset_path, "terabyte_processed_train.bin")
    test_dataset_bin = os.path.join(dataset_path, "terabyte_processed_test.bin")

    if dataset_type == 'bin':
        dataset_train = CriteoBinDataset(train_dataset_bin, batch_size=batch_size, shuffle=shuffle)
        dataset_test = CriteoBinDataset(test_dataset_bin, batch_size=test_batch_size)
    elif dataset_type == 'memmap':
        dataset_train = CriteoMemmapDataset(train_dataset_bin, batch_size=batch_size, shuffle=shuffle)
        dataset_test = CriteoMemmapDataset(test_dataset_bin, batch_size=test_batch_size)
    elif dataset_type == 'dist':
        dataset_train = DistCriteoDataset(
            os.path.join(dataset_path, "train"), batch_size=batch_size, shuffle=shuffle, **kwargs)

        if hasattr(dataset_train, 'num_samples'):
            mlperf_logger.log_event(key=mlperf_logger.constants.TRAIN_SAMPLES,
                                    value=dataset_train.num_samples)

        dataset_test = DistCriteoDataset(
            os.path.join(dataset_path, "test"), batch_size=test_batch_size, **kwargs)

        if hasattr(dataset_test, 'num_samples'):
            mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_SAMPLES,
                                    value=dataset_test.num_samples)

    data_loader_args = dict(
        batch_size=None,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=functools.partial(data_collate_fn, device=return_device, orig_stream=torch.cuda.current_stream()))
    data_loader_train = torch.utils.data.DataLoader(dataset_train, **data_loader_args)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, **data_loader_args)

    return data_loader_train, data_loader_test


def _dist_permutation(size):
    """Generate permutation for dataset shuffle

    Args:
        size (int): Size and high value of permutation

    Returns:
        permutation (ndarray):
    """
    if dist.get_world_size() > 1:
        # To guarantee all ranks have the same same permutation, generating it from rank 0 and sync
        # to other rank by writing to disk
        permutation_file = "/tmp/permutation.npy"
        if dist.get_local_rank() == 0:
            np.save(permutation_file, np.random.permutation(size))
        torch.distributed.barrier()
        permutation = np.load(permutation_file)
    else:
        permutation = np.random.permutation(size)

    return permutation


class CriteoBinDataset(Dataset):
    """Binary version of criteo dataset.

    Main structure is copied from reference. With following changes:
    - Removed unnecessary things, like counts_file which is not really used in training.
    - _transform_features is removed, doing it on GPU is much faster.

    """
    def __init__(self, data_file, batch_size=1, bytes_per_feature=4, shuffle=False):
        # dataset. single target, 13 dense features, 26 sparse features
        self.tad_fea = 1 + 13
        self.tot_fea = 1 + 13 + 26

        self.batch_size = batch_size
        self.bytes_per_batch = (bytes_per_feature * self.tot_fea * batch_size)

        self.num_batches = math.ceil(os.path.getsize(data_file) / self.bytes_per_batch)

        print('data file:', data_file, 'number of batches:', self.num_batches)
        self.file = open(data_file, 'rb', buffering=0)

        if shuffle:
            self.permutation = _dist_permutation(self.num_batches - 1)
        else:
            self.permutation = None

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.permutation is not None and idx != self.num_batches - 1:
            idx = self.permutation[idx]
        self.file.seek(idx * self.bytes_per_batch, 0)
        raw_data = self.file.read(self.bytes_per_batch)
        array = np.frombuffer(raw_data, dtype=np.int32)
        tensor = torch.from_numpy(array).view((-1, self.tot_fea))

        return tensor

    def __del__(self):
        self.file.close()


class CriteoMemmapDataset(Dataset):
    """Memmap version of criteo dataset

    Accessing sequentially is a lot faster on memmap

    Args:
        data_file (str): Full path to binary file of dataset
        batch_size (int):
        bytes_per_feature (int): Default 4
        shuffle (bool): If True, shuffle batch order by creating a permutation. Default False

    """
    def __init__(self, data_file, batch_size, bytes_per_feature=4, shuffle=False):
        self.record_width = 40  # 13 numerical, 26 categorical, 1 label
        self.batch_size = batch_size

        bytes_per_batch = (bytes_per_feature * self.record_width * batch_size)
        self.num_batches = math.ceil(os.path.getsize(data_file) / bytes_per_batch)

        if shuffle:
            self.permutation = _dist_permutation(self.num_batches - 1)
        else:
            self.permutation = None

        self.mmap = np.memmap(data_file, dtype=np.int32, mode='r')

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.permutation is not None and idx != self.num_batches - 1:
            idx = self.permutation[idx]
        start_idx = idx * (self.batch_size * self.record_width)
        end_idx = min((idx + 1) * (self.batch_size * self.record_width), self.mmap.shape[0])
        array = self.mmap[start_idx:end_idx]
        tensor = torch.tensor(array).reshape(-1, self.record_width)

        return tensor


class DistCriteoDataset(Dataset):
    """Distributed version of Criteo dataset

    Args:
        data_path (str): Full path to split binary file of dataset. It must contain numerical.bin, label.bin and
            cat_0 ~ cat_25.bin
        batch_size (int):
        shuffle (boolean):
        numerical_features(boolean): If True, load numerical features for bottom_mlp. Default False
        categorical_features (list or None): categorical features used by the rank
        prefetch_depth (int): How many samples to prefetch. Default 10.
    """
    def __init__(self, data_path, batch_size=1, shuffle=False, numerical_features=False, categorical_features=None,
                 prefetch_depth=10):

        bytes_per_label = 4
        self.bytes_per_batch = {
            "label": bytes_per_label * batch_size,
            "numerical": 13 * 4 * batch_size if numerical_features else 0,
            "categorical": 4 * batch_size if categorical_features is not None else 0
        }
        self.batch_size = batch_size

        self.label_file = os.open(os.path.join(data_path, F"label.bin"), os.O_RDONLY)

        label_file_size = os.fstat(self.label_file).st_size
        self.num_samples = int(label_file_size / bytes_per_label)
        self.num_batches = math.ceil(label_file_size / self.bytes_per_batch["label"])

        if numerical_features:
            self.numerical_features_file = os.open(os.path.join(data_path, "numerical.bin"), os.O_RDONLY)
            if math.ceil(os.fstat(self.numerical_features_file).st_size /
                         self.bytes_per_batch["numerical"]) != self.num_batches:
                raise ValueError("Size miss match in data files")
        else:
            self.numerical_features_file = None
        if categorical_features is not None and categorical_features:
            self.categorical_features_files = []
            for cat_id in categorical_features:
                cat_file = os.open(os.path.join(data_path, F"cat_{cat_id}.bin"), os.O_RDONLY)
                if math.ceil(
                        os.fstat(cat_file).st_size / self.bytes_per_batch["categorical"]) != self.num_batches:
                    raise ValueError("Size miss match in data files")
                self.categorical_features_files.append(cat_file)
        else:
            self.categorical_features_files = None

        if shuffle:
            self.permutation = _dist_permutation(self.num_batches - 1)
        else:
            self.permutation = None

        self.prefetch_depth = min(prefetch_depth, self.num_batches)
        self.prefetch_queue = queue.Queue()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def __len__(self):
        return self.num_batches

    def getitem(self, idx):
        if self.permutation is not None and idx != self.num_batches - 1:
            idx = self.permutation[idx]
        raw_label_data = os.pread(
            self.label_file, self.bytes_per_batch["label"], idx * self.bytes_per_batch["label"])
        click = torch.from_numpy(np.frombuffer(raw_label_data, dtype=np.float32))

        if self.numerical_features_file is not None:
            raw_numerical_data = os.pread(
                self.numerical_features_file,
                self.bytes_per_batch["numerical"],
                idx * self.bytes_per_batch["numerical"])
            numerical_features = torch.from_numpy(np.frombuffer(raw_numerical_data,
                                                                dtype=np.float32)).view(-1, 13)
        else:
            numerical_features = None

        if self.categorical_features_files is not None:
            categorical_features = []
            for cat_file in self.categorical_features_files:
                raw_cat_data = os.pread(
                    cat_file,
                    self.bytes_per_batch["categorical"],
                    idx * self.bytes_per_batch["categorical"])
                categorical_features.append(torch.from_numpy(np.frombuffer(raw_cat_data, dtype=np.int32)).unsqueeze(1))
            categorical_features = torch.cat(categorical_features, dim=1)
        else:
            categorical_features = None

        return click, numerical_features, categorical_features

    def __getitem__(self, idx):
        if self.prefetch_depth <= 1:
            return self.getitem(idx)

        if idx == 0:
            # Prefetch triggers MLperf timer. So start prefetch on first iter instead of in constructor.
            for i in range(self.prefetch_depth):
                self.prefetch_queue.put(self.executor.submit(self.getitem, (i)))
        if idx < self.num_batches - self.prefetch_depth:
            self.prefetch_queue.put(self.executor.submit(self.getitem, (idx + self.prefetch_depth)))
        return self.prefetch_queue.get().result()


    def __del__(self):
        os.close(self.label_file)
        if self.numerical_features_file is not None:
            os.close(self.numerical_features_file)
        if self.categorical_features_files is not None:
            for cat_file in self.categorical_features_files:
                os.close(cat_file)


def data_collate_fn(batch_data, device="cuda", orig_stream=None):
    """Split raw batch data to features and labels

    Args:
        batch_data (Tensor): One batch of data from CriteoBinDataset.
        device (torch.device): Output device. If device is GPU, split data on GPU is much faster.
        orig_stream (torch.cuda.Stream): CUDA stream that data processing will be run in.

    Returns:
        numerical_features (Tensor):
        categorical_features (Tensor):
        click (Tensor):
    """
    if not isinstance(batch_data, torch.Tensor):
        # Distributed pass
        if batch_data[1] is not None:
            numerical_features = torch.log(batch_data[1].to(device, non_blocking=True) + 1.).squeeze()
        else:
            # There are codes rely on numerical_features' dtype
            numerical_features = torch.empty(batch_data[0].shape[0], 13, dtype=torch.float32, device=device)
        if batch_data[2] is not None:
            categorical_features = batch_data[2].to(device, non_blocking=True)
        else:
            categorical_features = None
        click = batch_data[0].to(device, non_blocking=True).squeeze()
    else:
        batch_data = batch_data.to(device, non_blocking=True).split([1, 13, 26], dim=1)
        numerical_features = torch.log(batch_data[1].to(torch.float32) + 1.).squeeze()
        categorical_features = batch_data[2].to(torch.long)
        click = batch_data[0].to(torch.float32).squeeze()

    # record_stream() prevents data being unintentionally reused. Aslo NOTE that it may not work
    # with num_works >=1 in the DataLoader when use this data_collate_fn() as collate function.
    if orig_stream is not None:
        numerical_features.record_stream(orig_stream)
        if categorical_features is not None:
            categorical_features.record_stream(orig_stream)
        click.record_stream(orig_stream)

    return numerical_features, categorical_features, click


def prefetcher(load_iterator, prefetch_stream):
    def _prefetch():
        with torch.cuda.stream(prefetch_stream):
            try:
                data_batch = next(load_iterator)
            except StopIteration:
                return None

        return data_batch

    next_data_batch = _prefetch()

    while next_data_batch is not None:
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        data_batch = next_data_batch
        next_data_batch = _prefetch()
        yield data_batch

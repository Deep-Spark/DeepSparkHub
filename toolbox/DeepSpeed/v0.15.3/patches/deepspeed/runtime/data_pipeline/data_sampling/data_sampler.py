# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
coding=utf-8
 Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
Part of this code was adopted from https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/data_samplers.py
"""

import torch
import os
import numpy as np

import deepspeed.comm as dist
from deepspeed.utils import logger
from deepspeed.accelerator import get_accelerator
from ..constants import *
from ..curriculum_scheduler import CurriculumScheduler
from .indexed_dataset import MMapIndexedDataset
from .utils import create_mmap_dataset_builder, close_mmap_dataset_builder, find_fit_int_dtype

from torch.utils.data import Dataset, Sampler
from typing import Iterator, List, Optional

class DeepSpeedDataSampler(object):

    def __init__(self,
                 data_efficiency_config,
                 one_epoch_total_samples,
                 micro_batch_size,
                 data_parallel_rank,
                 data_parallel_size,
                 data_parallel_group,
                 gradient_accumulation_steps,
                 global_rank,
                 drop_last=True):
        # Keep a copy of input params for later use.
        self.data_efficiency_config = data_efficiency_config
        self.one_epoch_total_samples = one_epoch_total_samples
        self.index_dtype = find_fit_int_dtype(0, one_epoch_total_samples)
        self.total_samples = one_epoch_total_samples * self.data_efficiency_config[DATA_SAMPLING][
            DATA_SAMPLING_NUM_EPOCHS]
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_group = data_parallel_group
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_batch_size = self.micro_batch_times_data_parallel_size * \
            self.gradient_accumulation_steps
        self.global_rank = global_rank
        self.drop_last = drop_last
        self.np_rng = np.random.default_rng(self.data_efficiency_config[DATA_EFFICIENCY_SEED])
        self.state = {}
        self.batch = []
        self.consumed_samples = 0
        if self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]:
            self.curriculum_step = 0
            self.current_difficulties = {}
            self.data_cluster_paths = []
            self.data_cluster_current_position = []
            self.curriculum_schedulers = {}
            self.curriculum_index_to_sample = {}
            self.curriculum_index_to_metric = {}
            self.difficulty_type = {}
            self.clustering_type = {}
            self.data_1epoch_size = None
            if self.global_rank == 0:
                self.data_clusters = []
                self.data_cluster_sizes = []
                cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
                    CURRICULUM_LEARNING_CLUSTER_PATH]
                if not os.path.exists(cluster_path):
                    os.makedirs(cluster_path)
            for metric in self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS]:
                self.curriculum_schedulers[metric] = CurriculumScheduler(
                    data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][metric])
                self.difficulty_type[metric] = data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
                    CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_DIFFICULTY_TYPE]
                self.clustering_type[metric] = data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
                    CURRICULUM_LEARNING_METRICS][metric][CURRICULUM_LEARNING_CLUSTERING_TYPE]
                if self.global_rank == 0:
                    if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        self.curriculum_index_to_sample[metric] = MMapIndexedDataset(
                            data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS]
                            [metric][CURRICULUM_LEARNING_SAMPLE_PATH],
                            skip_warmup=True)
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            self.curriculum_index_to_metric[metric] = MMapIndexedDataset(
                                data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS]
                                [metric][CURRICULUM_LEARNING_METRIC_PATH],
                                skip_warmup=True)

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def set_custom_curriculum_learning_schedule(self, schedule_func_dict):
        for metric in self.curriculum_schedulers:
            if metric in schedule_func_dict:
                self.curriculum_schedulers[metric].set_custom_get_difficulty(schedule_func_dict[metric])

    def get_start_end_idx(self, batch_len=None):
        """
        given the length of a minibatch (defaults to micro-batch size * data_parallel_size),
        return the start and end indices of the current data parallel rank
        """
        batch_len = batch_len or self.micro_batch_times_data_parallel_size
        start_idx_fn = lambda r: round(r * batch_len / self.data_parallel_group.size())
        start_idx = start_idx_fn(self.data_parallel_rank)
        end_idx = start_idx_fn(self.data_parallel_rank + 1)
        return start_idx, end_idx

    def get_sample_based_on_metric_value(self, metric, value_start, value_end):
        new_samples = None
        for row in range(len(self.curriculum_index_to_sample[metric])):
            if self.curriculum_index_to_metric[metric][row] <= value_end and self.curriculum_index_to_metric[metric][
                    row] > value_start:
                row_samples = np.copy(self.curriculum_index_to_sample[metric][row])
                new_samples = row_samples if new_samples is None else np.concatenate(
                    (new_samples, row_samples), axis=None)
        return new_samples

    def get_sample_based_on_metric_percentile(self, metric, percentile_start, percentile_end):
        new_samples = None
        if self.data_1epoch_size is None:
            self.data_1epoch_size = sum(len(x) for x in self.curriculum_index_to_sample[metric])
        max_percentile = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_METRICS][
            metric][CURRICULUM_LEARNING_MAX_DIFFICULTY]
        sample_per_percentile = self.data_1epoch_size // max_percentile
        start_count = sample_per_percentile * percentile_start
        end_count = sample_per_percentile * percentile_end
        if percentile_end == max_percentile:
            end_count = self.data_1epoch_size
        current_count = 0
        for row in range(len(self.curriculum_index_to_sample[metric])):
            row_size = len(self.curriculum_index_to_sample[metric][row])
            if current_count + row_size > start_count:
                row_start = max(0, start_count - current_count)
                if current_count + row_size <= end_count:
                    row_end = row_size
                else:
                    row_end = end_count - current_count
                row_samples = np.copy(self.curriculum_index_to_sample[metric][row][row_start:row_end])
                new_samples = row_samples if new_samples is None else np.concatenate(
                    (new_samples, row_samples), axis=None)
            current_count += row_size
            if current_count >= end_count:
                break
        return new_samples

    def get_new_cluster(self, previous_difficulties):
        cluster_fname = CURRICULUM_LEARNING_CLUSTER_PREFIX
        for metric in self.curriculum_schedulers:
            cluster_fname = f"{cluster_fname}_{metric}{self.current_difficulties[metric]}"
        cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
            CURRICULUM_LEARNING_CLUSTER_PATH]
        cluster_path = f"{cluster_path}/{cluster_fname}"
        if self.global_rank == 0:
            new_cluster = None
            need_clustering = 0
            for metric in self.clustering_type:
                if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                    need_clustering += 1
            if need_clustering > 1:
                for metric in self.curriculum_schedulers:
                    if self.clustering_type[metric] == CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        metric_cluster = np.arange(start=0,
                                                   stop=self.one_epoch_total_samples,
                                                   step=1,
                                                   dtype=self.index_dtype)
                    else:
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            metric_cluster = self.get_sample_based_on_metric_value(metric, float('-inf'),
                                                                                   self.current_difficulties[metric])
                        elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                            metric_cluster = self.get_sample_based_on_metric_percentile(
                                metric, 0, self.current_difficulties[metric])
                    new_cluster = metric_cluster if new_cluster is None else \
                        np.intersect1d(new_cluster, metric_cluster, assume_unique=True)
                for cluster in self.data_clusters:
                    new_cluster = np.setdiff1d(new_cluster, cluster[0], assume_unique=True)
            else:
                if len(self.data_clusters) == 0:
                    new_cluster = np.arange(start=0, stop=self.one_epoch_total_samples, step=1, dtype=self.index_dtype)
                for metric in self.curriculum_schedulers:
                    if self.clustering_type[metric] != CURRICULUM_LEARNING_SINGLE_CLUSTER:
                        if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                            new_cluster = self.get_sample_based_on_metric_value(metric, previous_difficulties[metric],
                                                                                self.current_difficulties[metric])
                        elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                            new_cluster = self.get_sample_based_on_metric_percentile(
                                metric, previous_difficulties[metric], self.current_difficulties[metric])
            if new_cluster is not None and len(new_cluster) > 0:
                logger.info(
                    f"new data cluster (previous_difficulties {previous_difficulties}, current_difficulties {self.current_difficulties}) with size {len(new_cluster)} generated."
                )
                self.np_rng.shuffle(new_cluster)
                cluster_builder = create_mmap_dataset_builder(cluster_path, self.index_dtype)
                cluster_builder.add_item_numpy(new_cluster)
                close_mmap_dataset_builder(cluster_builder, cluster_path)
                self.data_clusters.append(MMapIndexedDataset(cluster_path, skip_warmup=True))
                self.data_cluster_sizes.append(len(self.data_clusters[-1][0]))
            else:
                logger.info(
                    f"new data cluster (previous_difficulties {previous_difficulties}, current_difficulties {self.current_difficulties}) has no matched data thus skipped."
                )
        dist.barrier(group=self.data_parallel_group)
        if os.path.isfile(f"{cluster_path}.bin"):
            self.data_cluster_paths.append(cluster_fname)
            self.data_cluster_current_position.append(0)

    def sample_from_clusters(self):
        num_clusters = len(self.data_clusters)
        weight_sum = sum(self.data_cluster_sizes)
        weights = [x / weight_sum for x in self.data_cluster_sizes]
        samples = self.np_rng.choice(num_clusters, self.global_batch_size, replace=True, p=weights)
        samples = np.bincount(samples, minlength=num_clusters)
        return samples

    def reshuffle_clusters(self, cidx):
        cluster_fname = self.data_cluster_paths[cidx]
        cluster_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
            CURRICULUM_LEARNING_CLUSTER_PATH]
        cluster_path = f"{cluster_path}/{cluster_fname}"
        cluster = np.copy(self.data_clusters[cidx][0])
        self.np_rng.shuffle(cluster)
        cluster_builder = create_mmap_dataset_builder(cluster_path, self.index_dtype)
        cluster_builder.add_item_numpy(cluster)
        close_mmap_dataset_builder(cluster_builder, cluster_path)
        self.data_clusters[cidx] = MMapIndexedDataset(cluster_path, skip_warmup=True)

    def get_sample_from_cluster(self, cidx, num_samples):
        start_idx = self.data_cluster_current_position[cidx]
        samples = list(np.copy(self.data_clusters[cidx][0][start_idx:(start_idx + num_samples)]))
        self.data_cluster_current_position[cidx] += num_samples
        if len(samples) < num_samples:
            num_samples_remained = num_samples - len(samples)
            logger.info(f"reshuffling cluster {cidx}.")
            self.reshuffle_clusters(cidx)
            samples += list(np.copy(self.data_clusters[cidx][0][:num_samples_remained]))
            self.data_cluster_current_position[cidx] = num_samples_remained
        return samples

    def get_next_global_batch(self):
        if self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][CURRICULUM_LEARNING_ENABLED]:
            self.curriculum_step += 1
            new_cluster = False
            previous_difficulties = {}
            for metric in self.curriculum_schedulers:
                next_difficulty = self.curriculum_schedulers[metric].update_difficulty(self.curriculum_step)
                if metric not in self.current_difficulties or \
                    next_difficulty != self.current_difficulties[metric]:
                    new_cluster = True
                if metric in self.current_difficulties:
                    previous_difficulties[metric] = self.current_difficulties[metric]
                else:
                    if self.difficulty_type[metric] == CURRICULUM_LEARNING_VALUE_BASED:
                        previous_difficulties[metric] = float('-inf')
                    elif self.difficulty_type[metric] == CURRICULUM_LEARNING_PERCENTILE_BASED:
                        previous_difficulties[metric] = 0
                self.current_difficulties[metric] = next_difficulty
            if new_cluster:
                self.get_new_cluster(previous_difficulties)
            if self.global_rank == 0:
                samples_per_cluster = self.sample_from_clusters()
                batch = []
                for cidx in range(len(samples_per_cluster)):
                    batch += self.get_sample_from_cluster(cidx, samples_per_cluster[cidx])
                self.np_rng.shuffle(batch)

                # broadcast tensor must have same shape across participants. So we fill batch with -1s when not full
                assert len(batch) <= self.global_batch_size
                batch += [-1] * (self.global_batch_size - len(batch))
                batch = torch.tensor(batch, device=get_accelerator().current_device_name(), dtype=torch.long).view(-1)
            else:
                batch = torch.empty(self.global_batch_size,
                                    device=get_accelerator().current_device_name(),
                                    dtype=torch.long)
            dist.broadcast(batch, 0, group=self.data_parallel_group)
            batch = batch[batch != -1]  # remove trailing -1s used to fill incomplete batch tensor
            self.batch = batch.tolist()

    def __iter__(self):
        while self.consumed_samples <= self.total_samples:
            if len(self.batch) == 0:
                self.get_next_global_batch()
            current_batch = self.batch[:self.micro_batch_times_data_parallel_size]
            self.batch = self.batch[self.micro_batch_times_data_parallel_size:]
            if len(current_batch) == self.micro_batch_times_data_parallel_size or \
                (len(current_batch) > 0 and not self.drop_last):
                start_idx, end_idx = self.get_start_end_idx(len(current_batch))
                yield current_batch[start_idx:end_idx]
                self.consumed_samples += len(current_batch)
                current_batch = []

    def state_dict(self):
        return {
            CURRICULUM_LEARNING_BATCH: self.batch,
            CURRICULUM_LEARNING_CONSUMED_SAMPLES: self.consumed_samples,
            CURRICULUM_LEARNING_STEP: self.curriculum_step,
            CURRICULUM_LEARNING_CURRENT_DIFFICULTIES: self.current_difficulties,
            CURRICULUM_LEARNING_DATA_CLUSTER_PATHS: self.data_cluster_paths,
            CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION: self.data_cluster_current_position,
            CURRICULUM_LEARNING_NP_RNG_STATE: np.random.get_state()
        }

    def load_state_dict(self, state_dict):
        self.batch = state_dict[CURRICULUM_LEARNING_BATCH]
        self.consumed_samples = state_dict[CURRICULUM_LEARNING_CONSUMED_SAMPLES]
        self.curriculum_step = state_dict[CURRICULUM_LEARNING_STEP]
        self.current_difficulties = state_dict[CURRICULUM_LEARNING_CURRENT_DIFFICULTIES]
        self.data_cluster_paths = state_dict[CURRICULUM_LEARNING_DATA_CLUSTER_PATHS]
        self.data_cluster_current_position = state_dict[CURRICULUM_LEARNING_DATA_CLUSTER_CURRENT_POSITION]
        np.random.set_state(state_dict[CURRICULUM_LEARNING_NP_RNG_STATE])
        cluster_root_path = self.data_efficiency_config[DATA_SAMPLING][CURRICULUM_LEARNING][
            CURRICULUM_LEARNING_CLUSTER_PATH]
        # Backward compatibility: previously data_cluster_paths were stored as
        # absolute paths. Now we changed it to just the file name so that even
        # if user moved the cluster files, the checkpoint loading still works
        # as long as user set the correct new CURRICULUM_LEARNING_CLUSTER_PATH
        # in deepspeed json config.
        for idx in range(len(self.data_cluster_paths)):
            if '/' in self.data_cluster_paths[idx]:
                self.data_cluster_paths[idx] = self.data_cluster_paths[idx].split('/')[-1]
        if self.global_rank == 0:
            for cluster_fname in self.data_cluster_paths:
                cluster_path = f"{cluster_root_path}/{cluster_fname}"
                self.data_clusters.append(MMapIndexedDataset(cluster_path, skip_warmup=True))
                self.data_cluster_sizes.append(len(self.data_clusters[-1][0]))


class DsRandomSampler(Sampler):
    data_source: Optional[Dataset]
    replacement: bool

    def __init__(self, data_source: Optional[Dataset], replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None,
                 lengths: Optional[List[int]] = None, batch_size: Optional[int] = None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

        if lengths is None:
            model_input_name = "input_ids"
            lengths = [len(feature[model_input_name]) for feature in self.data_source]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.batch_size = batch_size
        self.trs_random = os.getenv("RANDOM_SAMPLER", "LargerLengthsPre").upper()

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from self.group_larger_lengths_indices(torch.randperm(n, generator=generator).tolist())
            yield from self.group_larger_lengths_indices(torch.randperm(n, generator=generator).tolist()[:self.num_samples % n])

    def __len__(self) -> int:
        return self.num_samples

    def group_larger_lengths_indices(self, ori_indices: List[int] = []) -> List[int]:
        if len(ori_indices) == 0:
            return ori_indices

        limit_shape = int(os.getenv("LIMIT_SHAPE", 500))
        normal_indices = []
        larger_indices = []
        for i in range(len(ori_indices)):
            if self.lengths[ori_indices[i]] > limit_shape:
                larger_indices.append(ori_indices[i])
            else:
                normal_indices.append(ori_indices[i])

        import random
        scaling_factor  = 2
        max_multiple_bs = 8
        if self.trs_random == "LARGERLENGTHSPRE":
            # ========== add first random ==========
            larger_indices_length = len(larger_indices)
            if larger_indices_length % self.batch_size != 0:
                multiple_bs = larger_indices_length // self.batch_size + 1
                multiple_bs = max(min(multiple_bs * scaling_factor, max_multiple_bs), multiple_bs)
                diff_normal = multiple_bs * self.batch_size - larger_indices_length
                larger_indices.extend(normal_indices[0:diff_normal])
                normal_indices = normal_indices[diff_normal:]
                random.shuffle(larger_indices)
                larger_indices.extend(normal_indices)
            else:
                random.shuffle(larger_indices)
                larger_indices.extend(normal_indices)
            return larger_indices
        elif self.trs_random == "LARGERLENGTHSPOST":
            # ========== add last random ==========
            larger_indices_length = len(larger_indices)
            normal_indices_length = len(normal_indices)
            if larger_indices_length % self.batch_size != 0:
                multiple_bs = larger_indices_length // self.batch_size + 1
                multiple_bs = max(min(multiple_bs * scaling_factor, max_multiple_bs), multiple_bs)
                diff_normal = multiple_bs * self.batch_size - larger_indices_length
                larger_indices.extend(normal_indices[normal_indices_length - diff_normal:])
                normal_indices = normal_indices[0:normal_indices_length - diff_normal]
                random.shuffle(larger_indices)
                normal_indices.extend(larger_indices)
            else:
                random.shuffle(larger_indices)
                normal_indices.extend(larger_indices)
            return normal_indices
        elif self.trs_random == "LARGERLENGTHSPOSTANDSORT":
            # ========== larger lengths add last and larger-> smaller sort: test largest limit module id
            sort_lengths, sort_indices = torch.sort(torch.tensor(self.lengths), descending=True)
            larger_mask    = sort_lengths > limit_shape
            larger_indices = sort_indices[larger_mask].tolist()
            norm_indices   = sort_indices[~larger_mask].tolist()
            norm_indices.extend(larger_indices)
            return norm_indices
        else:
            raise RuntimeError(f"{self.trs_random} is not support")

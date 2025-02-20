# Copyright (c) 2018, deepakn94, codyaustun, robieta. All rights reserved.
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
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import torch
import torch.utils.data
import pandas

from mlperf_compliance import mlperf_log


class CFTrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, train_fname, nb_neg):
        self._load_train_matrix(train_fname)
        self.nb_neg = nb_neg

        mlperf_log.ncf_print(key=mlperf_log.INPUT_STEP_TRAIN_NEG_GEN, value=nb_neg)
        mlperf_log.ncf_print(key=mlperf_log.INPUT_HP_SAMPLE_TRAIN_REPLACEMENT)

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            tmp = line.split('\t')
            return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]
        with open(train_fname, 'r') as file:
            data = list(map(process_line, file))
        self.nb_users = max(data, key=lambda x: x[0])[0] + 1
        self.nb_items = max(data, key=lambda x: x[1])[1] + 1

        self.data = list(filter(lambda x: x[2], data))
        self.mat = set(self.data)

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data)

    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0:
            idx = idx // (self.nb_neg + 1)
            return self.data[idx][0], self.data[idx][1], np.ones(1, dtype=np.float32)  # noqa: E501
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = torch.LongTensor(1).random_(0, self.nb_items).item()
            while (u, j) in self.mat:
                j = torch.LongTensor(1).random_(0, self.nb_items).item()
            return u, j, np.zeros(1, dtype=np.float32)


def load_test_ratings(fname):
    return pandas.read_csv(fname, sep='\t', header=None).values

def load_train_ratings(fname):
    return pandas.read_csv(fname, sep='\t', header=None).values

def load_test_negs(fname):
    return pandas.read_csv(fname, sep='\t', header=None).values

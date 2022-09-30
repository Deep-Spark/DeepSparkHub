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

import os
import json
from functools import reduce
import time

class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
    def __exit__(self, *args):
        self.end = time.time()
        print("{} : {}".format(self.name, self.end - self.start))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    c = map(lambda p: reduce(lambda x, y: x * y, p.size()), model.parameters())
    return sum(c)


def save_config(config, run_dir):
    path = os.path.join(run_dir, "config_{}.json".format(config['timestamp']))
    with open(path, 'w') as config_file:
        json.dump(config, config_file)
        config_file.write('\n')


def save_result(result, path):
    write_heading = not os.path.exists(path)
    with open(path, mode='a') as out:
        if write_heading:
            out.write(",".join([str(k) for k, v in result.items()]) + '\n')
        out.write(",".join([str(v) for k, v in result.items()]) + '\n')

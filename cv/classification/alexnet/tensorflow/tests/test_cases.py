#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022-2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
import subprocess

import pytest

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
TRAIN_DIR = os.path.join(PROJECT_DIR, 'train_dir')
DATASETDIR = r'/mnt/datasets/imagenet_tfrecord/ILSVRC2012'


@pytest.fixture(scope="session", autouse=True)
def deploy_environment():
    """Deploy environment"""
    exec_cmd(
        'pip3 install absl-py '
        'git+https://github.com/NVIDIA/dllogger#egg=dllogger'
    )


@pytest.mark.timeout(3600)
def test_samples():
    """Test samples"""
    epochs = 1
    batch_size = 32
    optimizer = "momentum"
    cuda_visible_devices = '0'  # example '0,1,2,3,4,5,6,7'
    ix_num_cuda_visible_devices = len(cuda_visible_devices.split(","))
    env = {
        'TF_CUDNN_USE_AUTOTUNE': '1',
        'TF_CPP_MIN_LOG_LEVEL': '1',
        'CUDA_VISIBLE_DEVICES': cuda_visible_devices,
        'IX_NUM_CUDA_VISIBLE_DEVICES': str(ix_num_cuda_visible_devices)
    }

    python_cmd = (
        f"cd {PROJECT_DIR};"
        "python3 -u tf_cnn_benchmarks.py "
        f"--data_name=imagenet --data_dir={DATASETDIR} "
        f"--data_format=NCHW --batch_size={batch_size} "
        f"--model=alexnet --optimizer={optimizer} "
        f"--num_gpus={ix_num_cuda_visible_devices} "
        f"--weight_decay=1e-4 --train_dir={TRAIN_DIR} "
        "--eval_during_training_every_n_epochs=2 "
        f"--num_epochs={epochs} "
        "--num_eval_epochs=1 --datasets_use_caching "
        "--stop_at_top_1_accuracy=0.9 "
        "--num_intra_threads=1 --num_inter_threads=1"
    )
    if ix_num_cuda_visible_devices > 1:
        python_cmd += " --all_reduce_spec=pscpu "
        env['UMD_WAITAFTERLAUNCH'] = '1'

    return_code, _ = exec_cmd(python_cmd, env=env)

    assert return_code == 0


def exec_cmd(cmd, env=None):
    """Run cmd"""
    if env is None:
        env = os.environ
    else:
        env.update(os.environ)
    print(cmd)
    stdout_list = []
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=env) as process:
        while True:
            output = process.stdout.readline()
            if output:
                line_text = to_str(output)
                print(line_text, end='')
                stdout_list.append(line_text)
            if output == b'' and process.poll() is not None:
                break

    return process.returncode, stdout_list


def to_str(context):
    """Convert the context to str"""
    if isinstance(context, str):
        return context
    if isinstance(context, bytes):
        for coding in ['utf-8', 'gbk', "gb2312"]:
            try:
                res = context.decode(encoding=coding)
                return res
            except UnicodeDecodeError:
                pass

    return str(context)

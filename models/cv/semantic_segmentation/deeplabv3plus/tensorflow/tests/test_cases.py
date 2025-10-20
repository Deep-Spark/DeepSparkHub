#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
TRAIN_DIR = os.path.join(PROJECT_DIR, 'results')
DATASETDIR = r'/mnt/datasets/cityscapes'


@pytest.fixture(scope="session", autouse=True)
def deploy_environment():
    """Deploy environment"""
    exec_cmd(
        'pip3 install wandb;'
        'pip3 install urllib3==1.26.6'
    )


def test_samples():
    """Test samples"""
    epochs = 1
    batch_size = 8
    cuda_visible_devices = '0'  # example '0,1,2,3,4,5,6,7'
    ix_num_cuda_visible_devices = len(cuda_visible_devices.split(","))
    env = {
        'CUDA_VISIBLE_DEVICES': cuda_visible_devices,
    }
    if ix_num_cuda_visible_devices > 1:
        env['UMD_WAITAFTERLAUNCH'] = '1'
    python_cmd = (
        f"cd {PROJECT_DIR};"
        "python3 -u trainer.py cityscapes_resnet50 "
        f"--data-path {DATASETDIR} "
        f"--epochs {epochs} "
        f"--batch-size {batch_size} "
    )
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

# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

"""Module for training deeplabv3plus on cityscapes dataset."""

from glob import glob

import tensorflow as tf


# Sample Configuration
sample_configuration = {
    # We mandate specifying project_name and experiment_name in every config
    # file. They are used for wandb runs if wandb api key is specified.
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'cityscapes-segmentation-resnet-50-backbone',

    'train_dataset_config': {
        'images': sorted(glob('/path/to/cityscapes/leftImg8bit/train/*/*.png')),
        'labels': sorted(glob('/path/to/cityscapes/gtFine/train/*/*labelTrainIds.png')),
        'height': 512, 'width': 1024, 'batch_size': 8
    },

    'val_dataset_config': {
        'images': sorted(glob('/path/to/cityscapes/leftImg8bit/val/*/*.png')),
        'labels': sorted(glob('/path/to/cityscapes/gtFine/val/*/*labelTrainIds.png')),
        'height': 512, 'width': 1024, 'batch_size': 8
    },

    'num_classes': 19, 'backbone': 'resnet50', 'learning_rate': 0.0001,

    'checkpoint_dir': "./checkpoints/",
    'checkpoint_file_prefix': "deeplabv3plus_with_resnet50_",

    'epochs': 100
}


def CONFIG(args):
    config = sample_configuration.copy()
    if hasattr(args, 'data_path') and args.data_path:
        config['train_dataset_config']['images'] = sorted(
            glob(f'{args.data_path}/leftImg8bit/train/*/*.png'))
        config['train_dataset_config']['labels'] = sorted(
            glob(f'{args.data_path}/gtFine/train/*/*labelTrainIds.png'))
        config['val_dataset_config']['images'] = sorted(
            glob(f'{args.data_path}/leftImg8bit/val/*/*.png'))
        config['val_dataset_config']['labels'] = sorted(
            glob(f'{args.data_path}/gtFine/val/*/*labelTrainIds.png'))
    if hasattr(args, 'batch_size') and args.batch_size:
        config['train_dataset_config']['batch_size'] = args.batch_size
        config['val_dataset_config']['batch_size'] = args.batch_size
    if hasattr(args, 'epochs') and args.epochs:
        config['epochs'] = args.epochs

    return config

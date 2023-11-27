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

"""Module providing configuration for training on human parsing with resnet50
backbone"""

from glob import glob

import tensorflow as tf


sample_configuration = {
    'project_name': 'deeplabv3-plus',
    'experiment_name': 'human-parsing-resnet-50-backbone',

    'train_dataset_config': {
        'images': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Training/Images/*'
            )
        ),
        'labels': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Training/Category_ids/*'
            )
        ),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'val_dataset_config': {
        'images': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Validation/Images/*'
            )
        ),
        'labels': sorted(
            glob(
                './dataset/instance-level_human_parsing/'
                'instance-level_human_parsing/Validation/Category_ids/*'
            )
        ),
        'height': 512, 'width': 512, 'batch_size': 8
    },

    'num_classes': 20,
    'backbone': 'resnet50',
    'learning_rate': 0.0001,

    'checkpoint_dir': "./checkpoints/",
    'checkpoint_file_prefix':
    'deeplabv3-plus-human-parsing-resnet-50-backbone_',

    'epochs': 100
}


def CONFIG(args):
    config = sample_configuration.copy()
    if hasattr(args, 'data_path') and args.data_path:
        config['train_dataset_config']['images'] = sorted(
            glob(f'{args.data_path}/'
                 f'instance-level_human_parsing/Training/Images/*'))
        config['train_dataset_config']['labels'] = sorted(
            glob(f'{args.data_path}/'
                 f'instance-level_human_parsing/Training/Category_ids/*'))
        config['val_dataset_config']['images'] = sorted(
            glob(f'{args.data_path}/'
                 f'instance-level_human_parsing/Validation/Images/*'))
        config['val_dataset_config']['labels'] = sorted(
            glob(f'{args.data_path}/'
                 f'instance-level_human_parsing/Validation/Category_ids/*'))
    if hasattr(args, 'batch_size') and args.batch_size:
        config['train_dataset_config']['batch_size'] = args.batch_size
        config['val_dataset_config']['batch_size'] = args.batch_size
    if hasattr(args, 'epochs') and args.epochs:
        config['epochs'] = args.epochs

    return config

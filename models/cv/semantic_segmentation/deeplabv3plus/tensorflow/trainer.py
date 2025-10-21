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

"""Module for training deeplabv3plus on camvid dataset."""

# !pylint:disable=wrong-import-position

import argparse
from argparse import RawTextHelpFormatter

print("[-] Importing tensorflow...")
import tensorflow as tf  # noqa: E402
print(f"[+] Done! Tensorflow version: {tf.version.VERSION}")

print("[-] Importing Deeplabv3plus Trainer class...")
from deeplabv3plus.train import Trainer  # noqa: E402

print("[-] Importing config files...")
from config import CONFIG_MAP  # noqa: E402

if __name__ == "__main__":
    REGISTERED_CONFIG_KEYS = "".join(map(lambda s: f"  {s}\n", CONFIG_MAP.keys()))

    PARSER = argparse.ArgumentParser(
        description=f"""
            Runs DeeplabV3+ trainer with the given config setting.
            Registered config_key values:
            {REGISTERED_CONFIG_KEYS}""",
            formatter_class=RawTextHelpFormatter
        )
    PARSER.add_argument('config_key', 
                        help="Key to use while looking up "
                        "configuration from the CONFIG_MAP dictionary.")
    PARSER.add_argument("--wandb_api_key",
                        help="""Wandb API Key for logging run on Wandb.
                        If provided, checkpoint_dir is set to wandb://
                        (Model checkpoints are saved to wandb.)""",
                        default=None)
    PARSER.add_argument('--data-path', help='dataset')
    PARSER.add_argument('-b', '--batch-size', type=int)
    PARSER.add_argument('--epochs', type=int,
                        help='number of total epochs to run')
    ARGS = PARSER.parse_args()

    CONFIG = CONFIG_MAP[ARGS.config_key](ARGS)
    if ARGS.wandb_api_key is not None:
        CONFIG['wandb_api_key'] = ARGS.wandb_api_key
        CONFIG['checkpoint_dir'] = "wandb://"
    TRAINER = Trainer(CONFIG)
    HISTORY = TRAINER.train()

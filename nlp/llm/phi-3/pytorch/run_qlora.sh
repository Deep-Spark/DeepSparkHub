# Copyright (c) 2025, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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


export WANDB_DISABLED=True
torchrun --standalone --nproc_per_node 1 train.py \
                      --model_id microsoft/Phi-3-mini-4k-instruct \
                      --dataset_name iamtarun/python_code_instructions_18k_alpaca \
                      --use_4bit \
                      --bnb_4bit_use_double_quant \
                      --output_dir phi-3-mini-4k-instruct-qlora-alpaca

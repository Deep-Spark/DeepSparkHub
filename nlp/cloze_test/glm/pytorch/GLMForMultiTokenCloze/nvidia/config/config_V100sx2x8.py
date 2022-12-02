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
from training_event import DefaultTrainingEvent

train_batch_size = 4
eval_batch_size = 8

dist_backend = "nccl"

lr = 1e-5
weight_decay = 0.1
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_eps = 1e-08
gradient_accumulation_steps = 1
warmup = 0.1
lr_decay_ratio = 0.1
lr_decay_iters = 2169
log_freq = 0

training_event = DefaultTrainingEvent
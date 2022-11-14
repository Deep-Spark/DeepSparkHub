#!/bin/bash

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

cd PaddleNLP

pip3 install -r requirements.txt
python3 setup.py install --user

export CUDA_VISIBLE_DEVICES=0
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
python3 examples/language_model/bert/run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name SST2 \
    --max_seq_length 128 \
    --batch_size 32   \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --save_steps 500 \
    --output_dir ./tmp/ \
    --device gpu \
    --use_amp False

python3 examples/language_model/bert/export_model.py \
    --model_type bert \
    --model_path ./tmp/sst2_ft_model_6315.pdparams \
    --output_path ./infer_model/model

python3 examples/language_model/bert/predict.py \
    --model_path ./infer_model/model \
    --device gpu \
    --max_seq_length 128 \
    --device gpu 2>&1 | tee infer.log

cd ..
python3 train_bert.py
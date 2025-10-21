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
num_devices=`ixsmi --list-gpus | wc -l`

export TOKENIZERS_PARALLELISM=true


EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

python3 -m torch.distributed.launch --nproc_per_node=$num_devices --use_env \
train.py \
    --model_name_or_path t5-small \
    --do_train \
    --do_eval \
    --source_lang en \
    --target_lang de \
    --source_prefix "translate English to German: " \
    --dataset_name wmt16 \
    --dataset_config_name de-en \
    --output_dir /tmp/tst-translation \
    --per_device_train_batch_size=256 \
    --per_device_eval_batch_size=256 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_target_length=64 \
    --pad_to_max_length=True \
    --max_source_length=64 \
    --fp16=True \
    --fp16_backend=amp \
    --num_train_epochs 1 \
    --max_train_samples=900000 "$@";  check_status

exit ${EXIT_STATUS}
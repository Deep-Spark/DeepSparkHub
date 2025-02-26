#/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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

TP=4
PP=4

PROJ_HOME=$(dirname "$PWD")

## llama2-7B
python3 $PROJ_HOME/tools/checkpoint_util.py \
     --model-type GPT \
     --loader llama2_hf \
     --saver megatron \
     --target-tensor-parallel-size ${TP} \
     --target-pipeline-parallel-size ${PP} \
     --load-dir ./llama2-7b \
     --save-dir ./rlhf_llama2_7b_tp${TP}_pp${PP} \
     --tokenizer-model ./llama2-7b/tokenizer.model

## tinyllama-1.1B
python3 $PROJ_HOME/tools/checkpoint_util.py \
     --model-type GPT \
     --loader tinyllama_rlhf \
     --saver megatron \
     --target-tensor-parallel-size ${TP} \
     --target-pipeline-parallel-size ${PP} \
     --load-dir ./TinyLlama-1.1B \
     --save-dir ./rlhf_tinyllama_1.1b_tp${TP}_pp${PP} \
     --tokenizer-model ./TinyLlama-1.1B/tokenizer.model \
     --tinyllama \
     --custom-partition 5 5 6 6

mv ./rlhf_llama2_7b_tp${TP}_pp${PP}/iter_0000001/* ./rlhf_llama2_7b_tp${TP}_pp${PP}
mv ./rlhf_tinyllama_1.1b_tp${TP}_pp${PP}/iter_0000001/* ./rlhf_tinyllama_1.1b_tp${TP}_pp${PP}

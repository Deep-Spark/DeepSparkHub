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

## llama2-7B

pip3 install transformers urllib3==1.23 accelerate
python3 ../tools/checkpoint_util.py \
     --model-type GPT \
     --loader llama2_hf \
     --saver megatron \
     --target-tensor-parallel-size 4 \
     --target-pipeline-parallel-size 2 \
     --load-dir ./llama2-7b \
     --save-dir ./llama2_7b_megatron \
     --tokenizer-model ./llama2-7b/tokenizer.model


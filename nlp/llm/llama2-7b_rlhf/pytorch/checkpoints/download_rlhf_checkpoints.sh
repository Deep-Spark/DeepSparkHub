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

# wget http://sw.iluvatar.ai/download/apps/pretrained/nlp/RLHF/output_tinyLlama-1.1B.zip
# unzip output_tinyLlama-1.1B.zip
# rm -rf output_tinyLlama-1.1B.zip

# wget http://sw.iluvatar.ai/download/apps/pretrained/nlp/RLHF/output_step1_llama2_7b.zip
# unzip output_step1_llama2_7b.zip
# rm -rf output_step1_llama2_7b.zip

# wget http://sw.iluvatar.ai/download/apps/pretrained/nlp/RLHF/output_step1_llama2_7b_vocab_size_32000.zip
# unzip output_step1_llama2_7b_vocab_size_32000.zip
# rm -rf output_step1_llama2_7b_vocab_size_32000.zip	

wget http://sw.iluvatar.ai/download/apps/pretrained/nlp/llama2/llama2-7b.tar.gz
tar -xvf llama2-7b.tar.gz
rm -rf llama2-7b.tar.gz

wget http://sw.iluvatar.ai/download/apps/pretrained/nlp/RLHF/TinyLlama-1.1B-intermediate-step-240k-503b.zip
unzip TinyLlama-1.1B-intermediate-step-240k-503b.zip
rm -rf TinyLlama-1.1B-intermediate-step-240k-503b.zip
mv TinyLlama-1.1B-intermediate-step-240k-503b TinyLlama-1.1B
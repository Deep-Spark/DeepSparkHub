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
cd ../../../../../../data/model_zoo/
echo "check pretrained model..."
extract=1
pretrained_model_files=("bert_config.json"  "model.ckpt-28252.data-00000-of-00001"  "model.ckpt-28252.index"  "model.ckpt-28252.meta"  "vocab.txt")
for file in ${pretrained_model_files[*]}
do
    if [[ ! -f bert_pretrain_tf_ckpt/${file} ]]; then
        echo "bert_pretrain_tf_ckpt"/${file}" not exist"
        extract=0
    fi
done

if [[ $extract -eq 0 ]]; then
    tar zxvf bert_pretrain_ckpt_tf.tar.gz
fi

cd ../datasets
echo "check datasets..."
files=("bert_pretrain_tf_records/train_data/part-00015-of-00500" "bert_pretrain_tf_records/train_data/part-00014-of-00500" "bert_pretrain_tf_records/train_data/part-00013-of-00500" "bert_pretrain_tf_records/train_data/part-00012-of-00500" "bert_pretrain_tf_records/train_data/part-00011-of-00500" "bert_pretrain_tf_records/train_data/part-00010-of-00500" "bert_pretrain_tf_records/train_data/part-00009-of-00500" "bert_pretrain_tf_records/train_data/part-00008-of-00500" "bert_pretrain_tf_records/train_data/part-00007-of-00500" "bert_pretrain_tf_records/train_data/part-00006-of-00500" "bert_pretrain_tf_records/train_data/part-00005-of-00500" "bert_pretrain_tf_records/train_data/part-00004-of-00500" "bert_pretrain_tf_records/train_data/part-00003-of-00500" "bert_pretrain_tf_records/train_data/part-00002-of-00500" "bert_pretrain_tf_records/train_data/part-00000-of-00500" "bert_pretrain_tf_records/train_data/part-00001-of-00500"  "bert_pretrain_tf_records/eval_data/eval_tfrecord")
extract=1
for file in ${files[*]}
do
    if [[ ! -f ${file} ]]; then
        echo ${file}" not exist"
        extract=0
    fi
done
if [[ $extract -eq 0 ]]; then
    echo "tar zxvf bert_pretrain_tf_records.tar.gz"
    tar zxvf bert_pretrain_tf_records.tar.gz
fi


#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
EXIT_STATUS=0
check_status()
{
    if ((${PIPESTATUS[0]} != 0)); then
        EXIT_STATUS=1
    fi
}

horovodrun -np 16 --gloo python3 src/train_softmax_ddp.py \
        --logs_base_dir ./logs/facenet/ \
        --models_base_dir ./src/models/ \
        --data_dir ./data/webface_182_44 \
        --image_size 160 \
        --model_def models.inception_resnet_v1 \
        --lfw_dir ./data/lfw_data/lfw_160/ \
        --learning_rate -1 \
        --batch_size 128 \
        --optimizer ADAM \
        --max_nrof_epochs 500 \
        --keep_probability 0.8 \
        --random_flip \
        --random_crop \
        --use_fixed_image_standardization \
        --learning_rate_schedule_file ./data/learning_rate_schedule_classifier_casia_ddp.txt \
        --weight_decay 5e-4 \
        --embedding_size 512 \
        --lfw_distance_metric 1 \
        --lfw_use_flipped_images \
        --lfw_subtract_mean \
        --validation_set_split_ratio 0.01 \
        --validate_every_n_epochs 5 \
        --prelogits_norm_loss_factor 5e-4 \
        --gpu_memory_fraction 0.9 \
        --seed 43 \
        --epoch_size 200 "$@";  check_status

exit ${EXIT_STATUS}

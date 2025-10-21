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

set -x

pip3 install -r requirements.txt
pip uninstall numpy
pip install numpy==1.23.5
python3 setup.py develop

mkdir -p data
ln -s /mnt/deepspark/data/datasets/coco2017 data/coco
mkdir -p /root/.torch/models/
cp /mnt/deepspark/data/checkpoints/R-50.pkl /root/.torch/models/

timeout 1800 python3 -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train_net.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x
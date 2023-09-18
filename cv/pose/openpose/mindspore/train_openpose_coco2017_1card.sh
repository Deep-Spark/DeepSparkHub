#!/bin/bash

# Copyright (c) 2023,Shanghai Iluvatar CoreX Semiconductor Co.,Ltd.
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except incompliance with the License. You may obtain
# a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
pip3 install -r requirements.txt

bash scripts/run_standalone_train.sh /home/coco2017/train2017 /home/coco2017/annotations/person_keypoints_train2017.json /home/coco2017/ignore_mask_train /home/vgg19-0-97_5004.ckpt

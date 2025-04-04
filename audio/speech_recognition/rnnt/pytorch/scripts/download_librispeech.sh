#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



DATA_ROOT_DIR=$1
DATA_SET="LibriSpeech"
DATA_DIR="${DATA_ROOT_DIR}/${DATA_SET}"
# if [ ! -d "$DATA_DIR" ]
# then
#     mkdir $DATA_DIR
#     chmod go+rx $DATA_DIR
#     python utils/download_librispeech.py utils/librispeech.csv $DATA_DIR -e ${DATA_ROOT_DIR}/
# else
#     echo "Directory $DATA_DIR already exists."
# fi

chmod go+rx $DATA_DIR
python3 utils/download_librispeech.py utils/librispeech.csv $DATA_DIR -e ${DATA_ROOT_DIR}/

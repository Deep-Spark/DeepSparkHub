#!/bin/bash
# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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



LOG_DIR=$1

docker run -it --rm \
  --gpus='all' \
  --shm-size=4g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $LOG_DIR:/logdir \
  -v $LOG_DIR/datasets:/datasets \
  -v $PWD:/code \
  -v $PWD:/workspace/rnnt \
  mlperf/rnn_speech_recognition bash

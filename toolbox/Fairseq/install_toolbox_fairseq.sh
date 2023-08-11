#!/bin/bash
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

if [ -d "./fairseq" ]; then
  echo "Already installed Fairseq." 
  exit 0
fi
# clone fairseq
git clone -b v0.10.2 https://github.com/facebookresearch/fairseq.git
# apply patch
cp -r -T patch/ fairseq/
# install fairseq
cd fairseq/
pip3 install Cython
python3 setup.py develop
pip3 install sacrebleu==1.5.1 sacremoses

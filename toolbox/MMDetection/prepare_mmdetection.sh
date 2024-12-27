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

set -x

# mmdetection
MMDET_VERSION=$1

MMDET_VERSION=${MMDET_VERSION:-v2.22.0}

if [ -d "./mmdetection" ]; then
  echo "Already installed MMDetection." 
  exit 0
fi

echo "=====Prepare mmdetection ${MMDET_VERSION} START====="

git clone --depth 1 -b ${MMDET_VERSION} https://github.com/open-mmlab/mmdetection.git

# configs/rtmdet reference from https://github.com/open-mmlab/mmdetection/tree/fe3f809a0a514189baf889aa358c498d51ee36cd/configs/rtmdet which is first added in v3.0.0rc1
cp -r -T patch/mmdetection/ mmdetection/

cd mmdetection/
bash clean_mmdetection.sh
bash build_mmdetection.sh

# install pip requirements
pip3 install build_pip/mmdet-*+corex*-py3-none-any.whl
pip3 install yapf==0.31.0 
pip3 install addict 
pip3 install opencv-python

# install libGL
yum install -y mesa-libGL
echo "=====Prepare mmdetection ${MMDET_VERSION} END====="


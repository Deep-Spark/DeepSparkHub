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

cp -r -T patch/mmdetection/ mmdetection/

cd mmdetection/
bash clean_mmdetection.sh
bash build_mmdetection.sh

# install pip requirements
pip3 install build_pip/mmdet-*+corex*-py3-none-any.whl
pip3 install yapf addict opencv-python

# install libGL
yum install -y mesa-libGL
echo "=====Prepare mmdetection ${MMDET_VERSION} END====="


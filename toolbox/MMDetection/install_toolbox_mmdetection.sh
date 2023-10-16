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

if [ -d "./mmdetection" ]; then
  echo "Already installed MMDetection." 
  exit 0
fi


# mmcv
git clone --depth 1 -b v1.5.3 https://github.com/open-mmlab/mmcv.git

cp -r -T patch/mmcv/v1.5.3 mmcv/

cd mmcv

rm -rf mmcv/ops/csrc/common/cuda/spconv/ mmcv/ops/csrc/common/utils/spconv/
rm -f mmcv/ops/csrc/pytorch/cpu/sparse_*
rm -f mmcv/ops/csrc/pytorch/cuda/fused_spconv_ops_cuda.cu
rm -f mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu
rm -f mmcv/ops/csrc/pytorch/cuda/sparse_*
rm -f mmcv/ops/csrc/pytorch/sp*

bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh

# mmdetection
cd ../
git clone --depth 1 -b v2.22.0 https://github.com/open-mmlab/mmdetection.git

cp -r -T patch/mmdetection/ mmdetection/

cd mmdetection/
bash clean_mmdetection.sh
bash build_mmdetection.sh

# install pip requirements
pip3 install build_pip/mmdet-2.22.0+corex*-py3-none-any.whl
pip3 install yapf addict opencv-python

# install libGL
yum install -y mesa-libGL

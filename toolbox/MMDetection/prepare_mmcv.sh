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

# mmcv
MMCV_VERSION=$1

MMCV_VERSION=${MMCV_VERSION:-v1.5.3}

if [ -d "./mmcv/${MMCV_VERSION}" ]; then
  echo "mmcv ${MMCV_VERSION} already prepared."
  exit 0
fi

echo "=====Prepare mmcv ${MMCV_VERSION} START====="
mkdir -p mmcv/
pushd mmcv/
git clone --depth 1 -b ${MMCV_VERSION} https://github.com/open-mmlab/mmcv.git ${MMCV_VERSION}
popd

cp -r -T patch/mmcv/${MMCV_VERSION} mmcv/${MMCV_VERSION}/

cd mmcv/${MMCV_VERSION}/

rm -rf mmcv/ops/csrc/common/cuda/spconv/ mmcv/ops/csrc/common/utils/spconv/
rm -f mmcv/ops/csrc/pytorch/cpu/sparse_*
rm -f mmcv/ops/csrc/pytorch/cuda/fused_spconv_ops_cuda.cu
rm -f mmcv/ops/csrc/pytorch/cuda/spconv_ops_cuda.cu
rm -f mmcv/ops/csrc/pytorch/cuda/sparse_*
rm -f mmcv/ops/csrc/pytorch/sp*

echo "=====Prepare mmcv ${MMCV_VERSION} END====="

echo "=====Install mmcv ${MMCV_VERSION} START====="
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
echo "=====Install mmcv ${MMCV_VERSION} END====="


# Copyright (c) 2023, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.

git clone -b v0.3.0 https://github.com/hpcaitech/ColossalAI.git
cp -r -T patch/ ColossalAI/
cd ColossalAI
CUDA_EXT=1
pip3 install .
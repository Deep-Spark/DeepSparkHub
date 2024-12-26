# Copyright (c) 2022, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
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
set -euox pipefail


pip3 install argparse easydict h5py matplotlib numpy open3d==0.10 opencv-python pyyaml scipy tensorboardX timm==0.4.5  tqdm transforms3d termcolor scikit-learn==0.24.1 --default-timeout=1000

# Chamfer Distance
bash install.sh
# PointNet++
cd ./Pointnet2_PyTorch
pip3 install pointnet2_ops_lib/.
cd -
# GPU kNN
pip3 install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# prepare data

cd data/ShapeNet55-34

if [[ ! -f "ShapeNet55.zip" ]]; then
    wget http://files.deepspark.org.cn:880/deepspark/data/datasets/ShapeNet55.zip
    unzip ShapeNet55.zip
    mv ShapeNet55/shapenet_pc/ .
    rm -r ShapeNet55
fi

cd -

cd ./data/ModelNet/modelnet40_normal_resampled

if [[ ! -f "processed_ModelNet.zip" ]]; then
    wget http://files.deepspark.org.cn:880/deepspark/data/datasets/processed_ModelNet.zip
    unzip processed_ModelNet.zip
    mv processed_ModelNet/* .
fi


apt update
apt install libgl1-mesa-glx




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

GLM_DATA_DIR=$1

set -euox pipefail


if [ ! -n "$GLM_DATA_DIR" ]; then
  echo "set data dir to default"
  GLM_DATA_DIR=/home/data/perf/glm/
fi

echo "data save in "$GLM_DATA_DIR

mkdir -p ${GLM_DATA_DIR}
cd ${GLM_DATA_DIR}

if [[ ! -f "ReCoRD.zip" ]]; then
        echo "ReCoRD.zip not exist"
        echo "wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip"
        wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/ReCoRD.zip
fi

unzip -o ReCoRD.zip

cd -

pip3 install -r ./data_preprocessing/requirements.txt

mkdir -p ${GLM_DATA_DIR}/ReCoRD/glm_train_eval_hdf5_sparse/train_hdf5
mkdir -p ${GLM_DATA_DIR}/ReCoRD/glm_train_eval_hdf5_sparse/eval_hdf5

GLM_DATA_DIR=${GLM_DATA_DIR} python3 ./data_preprocessing/create_train_eval_data.py

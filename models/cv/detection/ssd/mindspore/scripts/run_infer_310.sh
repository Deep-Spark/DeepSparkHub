#!/bin/bash
# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATA_PATH] [DVPP] [CONFIG_PATH] [DEVICE_ID]
    DVPP is mandatory, and must choose from [DVPP|CPU], it's case-insensitive
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
data_path=$(get_real_path $2)
cfg_path=$4

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

echo "mindir name: "$model
echo "dataset path: "$data_path
echo "config path: " $cfg_path
echo "device id: "$device_id

export ASCEND_HOME=/usr/local/Ascend/
if [ -d ${ASCEND_HOME}/ascend-toolkit ]; then
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/ascend-toolkit/latest/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/ascend-toolkit/latest/atc/lib64:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export TBE_IMPL_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:${TBE_IMPL_PATH}:$ASCEND_HOME/ascend-toolkit/latest/fwkacllib/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/ascend-toolkit/latest/opp
else
    export ASCEND_HOME=/usr/local/Ascend/latest/
    export PATH=$ASCEND_HOME/fwkacllib/bin:$ASCEND_HOME/fwkacllib/ccec_compiler/bin:$ASCEND_HOME/atc/ccec_compiler/bin:$ASCEND_HOME/atc/bin:$PATH
    export LD_LIBRARY_PATH=$ASCEND_HOME/fwkacllib/lib64:/usr/local/lib:$ASCEND_HOME/atc/lib64:$ASCEND_HOME/acllib/lib64:$ASCEND_HOME/driver/lib64:$ASCEND_HOME/add-ons:$LD_LIBRARY_PATH
    export PYTHONPATH=$ASCEND_HOME/fwkacllib/python/site-packages:$ASCEND_HOME/atc/python/site-packages:$PYTHONPATH
    export ASCEND_OPP_PATH=$ASCEND_HOME/opp
fi

function compile_app()
{
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    image_shape=`cat ${cfg_path} | grep img_shape`
    height=${image_shape:12:3}
    width=${image_shape:17:3}

    ../ascend310_infer/out/main --mindir_path=$model --dataset_path=$data_path --cpu_dvpp=CPU --device_id=$device_id --image_height=$height --image_width=$width &> infer.log
}

function cal_acc()
{
    python ../postprocess.py --result_path=./result_Files --img_path=$data_path --config_path=${cfg_path} --drop=True &> acc.log &
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi
# !/bin/bash
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
# if [[ ${ID} == "ubuntu" ]]; then
#   echo ${ID}
#   #sudo apt-get install libprotobuf-dev protobuf-compiler
# elif [[ ${ID} == "centos" ]]; then
#   echo ${ID}
#   #yum install libprotobuf-dev protobuf-compiler
# else
#   echo "Unable to determine OS..."
# fi
# pip install scikit-build
# pip install -r requirements.txt
# HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITH_TENSORFLOW=1 pip install horovod[tensorflow]

: ${IX_NUM_CUDA_VISIBLE_DEVICES:=1}
: ${HOROVOD_LAUNCH_CMD:="horovodrun -np ${IX_NUM_CUDA_VISIBLE_DEVICES} --gloo"}
: ${BATCH_SIZE:=8}

bash ./run_get_hippocampus_data.sh
if [[ $? != 0 ]]; then
  echo "ERROR: get hippocampus data failed"
  exit 1
fi

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p ${LOG_DIR}
fi
DATE=`date +%Y%m%d%H%M%S`

${HOROVOD_LAUNCH_CMD} python3 examples/vnet_train_and_evaluate.py --gpus 1 --batch_size ${BATCH_SIZE} --base_lr 0.0001 --data_dir ./data/Task04_Hippocampus/ --model_dir ./model_train/ "$@"
if [[ $? != 0 ]]; then
  echo "ERROR: run vnet train and evaluate failed"
  exit 1
fi

exit 0

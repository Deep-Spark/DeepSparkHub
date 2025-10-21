#!/bin/bash
CUR_DIR=$(cd "$(dirname "$0")";pwd)
PROJECT_ROOT="${CUR_DIR}/../.."
DATASET_DIR="${PROJECT_ROOT}/data/datasets"
MODEL_CPT_DIR="${PROJECT_ROOT}/data/model_zoo/ssd_tf"
VOC_RECORD_DIR="${DATASET_DIR}/tf_ssd_voc_record"
SSD_ROOT="${PROJECT_ROOT}/../models/cv/detection/ssd/tensorflow"

# determine whether the user is root mode to execute this script
prefix_sudo=""
current_user=$(whoami)
if [ "$current_user" != "root" ]; then
    echo "User $current_user need to add sudo permission keywords"
    prefix_sudo="sudo"
fi

echo "prefix_sudo= $prefix_sudo"

# pip3 install --upgrade tf_slim
pip3 uninstall -y protobuf
pip3 install "protobuf<4.0.0"
source $(cd `dirname $0`; pwd)/../_utils/which_install_tool.sh
if command_exists apt; then
	$prefix_sudo apt install -y git numactl
elif command_exists dnf; then
	$prefix_sudo dnf install -y git numactl
else
	$prefix_sudo yum install -y git numactl
fi

# Prepare checkpoint
echo "Prepare SSD's checkpoint"
if [ -d "$MODEL_CPT_DIR" ]; then
    rm -rf $MODEL_CPT_DIR
fi
mkdir -p $MODEL_CPT_DIR

echo "Unarchive model checkpoint"
tar -xzvf "${MODEL_CPT_DIR}.tar" -C "${MODEL_CPT_DIR}/../"
if [ -d "$SSD_ROOT/model" ]; then
    rm "$SSD_ROOT/model"
fi
ln -s ${MODEL_CPT_DIR} "$SSD_ROOT/model"
echo "Make soft link from ${MODEL_CPT_DIR} to $SSD_ROOT/model"

# Prepare voc dataset
echo "Start make SSD's dataset"
if [ -d $VOC_RECORD_DIR ]; then
    rm -rf $VOC_RECORD_DIR
fi

mkdir $VOC_RECORD_DIR

cd $SSD_ROOT
python3 dataset/convert_voc_sample_tfrecords.py \
--dataset_directory=$DATASET_DIR \
--output_directory=$VOC_RECORD_DIR \
--train_splits VOC2012_sample \
--validation_splits VOC2012_sample

if [ -d "dataset/tfrecords" ]; then
    rm "dataset/tfrecords"
fi

ln -s $VOC_RECORD_DIR "./dataset/tfrecords"
echo "End make SSD's dataset"
cd $CUR_DIR

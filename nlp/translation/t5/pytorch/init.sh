#!/bin/bash
set -e

: ${MODEL_CHECKPOINT_DIR:="./"}
: ${DATASET_DIR:="./"}

if [ ! -d "./pretrained/t5_small" ];then
   tar zxvf "${MODEL_CHECKPOINT_DIR}/t5_small.tar.gz" -C ./pretrained
fi

if [ ! -d "./wmt14_data/wmt14-en-de-pre-processed" ];then
   tar zxvf "${DATASET_DIR}/wmt14-en-de-pre-processed.tar.gz" -C ./wmt14_data
fi

exit 0
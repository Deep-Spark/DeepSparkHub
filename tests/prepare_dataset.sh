# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.

PROJ_DIR=$(cd `dirname $0`; pwd)
DATASET_DIR="${PROJ_DIR}/data/datasets"
MODEL_ZOO_DIR="${PROJ_DIR}/data/model_zoo"


# Unarchive datas
if [ -f "datasets.tgz" ]; then
    tar -zxf "datasets.tgz"
fi

# Prepare coco
cd ${DATASET_DIR}/coco2017
if [[ -f "annotations_trainval2017.zip" ]]; then
    echo "Unarchive annotations_trainval2017.zip"
    unzip -q annotations_trainval2017.zip
fi
if [[ -f "train2017.zip" ]]; then
    if [[ ! -d "${DATASET_DIR}/coco2017/train2017" ]]; then
        echo "Unarchive train2017.zip"
        unzip -q train2017.zip
    fi
fi
if [[ -f "val2017.zip" ]]; then
    if [[ ! -d "${DATASET_DIR}/coco2017/val2017" ]]; then
        echo "Unarchive val2017.zip"
        unzip -q val2017.zip
    fi
fi
if [[ -f "val2017_mini.zip" ]]; then
    if [[ ! -d "${DATASET_DIR}/coco2017/val2017_mini" ]]; then
        echo "Unarchive val2017_mini.zip"
        unzip -q val2017_mini.zip
    fi
fi

cd ${DATASET_DIR}
# TGZS=`find . -iname 'CamVid*.tgz' -or -iname '*VOC*.tgz' -or -iname 'imagenet*.tgz' -or -iname 'coco*.tgz'`
TGZS=$(ls -al | grep -oE 'CamVid[^ ]*.tgz|[^ ]*VOC[^ ]*.tgz|imagenet[^ ]*.tgz|coco[^ ]*.tgz')
for path in $TGZS; do
    data_name=`echo "${path}" | cut -f2 -d'/' | cut -f1 -d'.'`
    if [[ -d "${data_name}" ]]; then
        echo "Skip ${path}"
        continue
    fi

    echo "Unarchive ${path}"
    cd ${path%/*}
    if [ -w "${path##*/}" ]; then
        echo "该文件有写入权限。"
    else
        echo "该文件没有写入权限。"
        continue
    fi
    tar zxf ${path##*/}
    cd ${DATASET_DIR}
done


# Prepare pretrained data
cd ${DATASET_DIR}/bert_mini
if [[ ! -d "${DATASET_DIR}/bert_mini/2048_shards_uncompressed" ]]; then
    echo "Unarchive 2048_shards_uncompressed_mini"
    tar -xzf 2048_shards_uncompressed_mini.tar.gz
fi
if [[ ! -d "${DATASET_DIR}/bert_mini/eval_set_uncompressed" ]]; then
    echo "Unarchive eval_set_uncompressed.tar.gz"
    tar -xzf eval_set_uncompressed.tar.gz
fi
cd ../../../


# Prepare model's checkpoint
if [ ! -d "${HOME}/.cache/torch/hub/checkpoints/" ]; then
    echo "Create checkpoints dir"
    mkdir -p ${HOME}/.cache/torch/hub/checkpoints/
fi


if [ -d "${MODEL_ZOO_DIR}" ]; then
    cd ${MODEL_ZOO_DIR}
    checkpoints=`find . -name '*.pth' -or -name '*.pt'`
    for cpt in $checkpoints; do
        if [[ ! -f "${HOME}/.cache/torch/hub/checkpoints/${cpt}" ]]; then
            echo "Copy $cpt to ${HOME}/.cache/torch/hub/checkpoints/"
            cp $cpt ${HOME}/.cache/torch/hub/checkpoints/
        fi
    done
fi

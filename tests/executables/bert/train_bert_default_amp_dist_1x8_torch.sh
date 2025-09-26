#!/bin/bash

# /***************************************************************************************************
# * Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# * Copyright Declaration: This software, including all of its code and documentation,
# * except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# * Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# * Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# * CoreX. No user of this software shall have any right, ownership or interest in this software and
# * any use of this software shall be in compliance with the terms and conditions of the End User
# * License Agreement.
#  **************************************************************************************************/

set -euox pipefail

source ../_utils/global_environment_variables.sh

: ${BATCH_SIZE:=10}

cd ../../nlp/language_model/bert_sample/pytorch/base
if [ "$?" != "0" ]; then
    echo "train status: fail."
    exit 1
fi


bash run_training.sh \
--name default \
--config V100x1x8 \
--data_dir ../../../../../../data/datasets/bert_mini/ \
--max_steps 500 \
--train_batch_size ${BATCH_SIZE} \
--target_mlm_accuracy 0.33 \
--init_checkpoint "../../../../../../data/datasets/bert_mini/model.ckpt-28252.apex.pt"

if [ "$?" != "0" ]; then
    echo "train status: fail."
    exit 1
fi

echo "train status: pass."

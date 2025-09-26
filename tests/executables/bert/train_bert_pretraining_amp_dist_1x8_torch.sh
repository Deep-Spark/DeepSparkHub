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

: ${BATCH_SIZE:=27}

cd ../../nlp/language_model/bert_sample/pytorch/base/
if [ "$?" != "0" ]; then
    echo "ERROR: ../../nlp/language_model/bert_sample/pytorch/base/ not exist."
    exit 1
fi

master_port=22233
bash run_training.sh --name iluvatar --config 03V100x1x8 --train_batch_size ${BATCH_SIZE} --data_dir ../../../../../../data/datasets/bert_mini/ --master_port $master_port
if [ "$?" != "0" ]; then
    echo "eval result: fail."
    exit 1
fi

echo "eval result: pass."

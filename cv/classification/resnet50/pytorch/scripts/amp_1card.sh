# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.
SCRIPT_DIR=$(cd `dirname $0`; pwd)
PROJECT_DIR=$SCRIPT_DIR/..
export CUDA_VISIBLE_DEVICES=0
bash $SCRIPT_DIR/train_resnet50_imagenet.sh --batch-size 300 --amp --dali $@

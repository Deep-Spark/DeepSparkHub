#!/bin/bash
WORK_PATH=$(dirname $(readlink -f $0))

set -e
cp ${WORK_PATH}/../../../../data/datasets/bert_scripts/*  .
exit 0
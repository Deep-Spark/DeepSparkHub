#!/bin/bash

: ${DATASET_DIR:="./"}


if [ ! -d "./imagenette" ]; then
    echo "Make soft link form ${DATASET_DIR} to tf_cnn_benckmarks"
    ln -s "${DATASET_DIR}/imagenette_tfrecord" imagenette
fi


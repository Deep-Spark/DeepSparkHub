#!/bin/bash

: ${DATA_DIR:="./"}


if [ ! -d "./imagenette" ]; then
    echo "Make soft link form ${DATA_DIR} to tf_cnn_benckmarks"
    ln -s "${DATA_DIR}/imagenette_tfrecord" imagenette
fi


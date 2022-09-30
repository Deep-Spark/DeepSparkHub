#!/bin/bash

echo "prepare data..."
DATA_URL=$1

mkdir -p data/kits19/train

data_file="data/kits19/train/kits19.tar.gz"

if [ ! -e ${data_file} ]; then
    echo "ERROR: Invalid data file ${data_file}"
    wget -P data/kits19/train ${DATA_URL}
fi

echo "Uncompressing the kits19!"
tar -xf ${data_file} -C data/kits19/train



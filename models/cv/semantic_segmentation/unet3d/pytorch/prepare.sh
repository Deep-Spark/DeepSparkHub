#!/bin/bash
apt update -y
apt install -y numactl git wget

# echo "install packages..."


#pip3 install "git+https://github.com/mlperf/logging.git@0.7.0"
#pip3 install -r requirements.txt
#python3 setup.py install

echo "prepare data..."

mkdir -p /home/datasets/cv/kits19/train

data_file="/home/datasets/cv/kits19/train/kits19.tar.gz"

if [ ! -e ${data_file} ]; then
    echo "ERROR: Invalid data file ${data_file}"
    # wget -P /home/datasets/cv/kits19/train /url/to/kits19.tar.gz
fi

echo "Uncompressing the kits19!"
tar -xf ${data_file} -C /home/datasets/cv/kits19/train



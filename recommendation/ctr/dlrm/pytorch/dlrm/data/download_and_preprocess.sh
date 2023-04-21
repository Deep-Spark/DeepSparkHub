#!/bin/bash


# Download files
mkdir -p /home/datasets/recommendation/Criteo_Terabyte
cd /home/datasets/recommendation/Criteo_Terabyte

echo 'download files ...'

if [ ! -f "day_0" ];then
echo 'download day_0 ...'
wget -c https://sacriteopcail01.z16.web.core.windows.net/day_0.gz
gzip -dk day_0.gz
else
echo "day_0 has already exist"
fi

if [ ! -f "day_1" ];then
echo 'download day_1 ...'
wget -c https://sacriteopcail01.z16.web.core.windows.net/day_1.gz
gzip -dk day_1.gz
else
echo "day_1 has already exist"
fi

if [ ! -f "day_2" ];then
echo 'download day_2 ...'
wget -c https://sacriteopcail01.z16.web.core.windows.net/day_2.gz
gzip -dk day_2.gz
else
echo "day_2 has already exist"
fi


filesize=`ls -l ./day_0 | awk '{ print $5 }'`
cd -
# # Extracted files
# # Into path modelzoo-research/recommendation/ctr/dlrm/pytorch/dlrm/data
# # If day_0, day_1, day_2 is bigger than 6 GB, it means these files have not been extracted.
echo "extract files ..."

if [ "$filesize" -gt 6000000000 ];then
python3 dlrm_data_pytorch.py --fun='extract' \
--raw-data-file=/home/datasets/recommendation/Criteo_Terabyte/day \
--extract-sample-rate=0.1 \
--extract-days="0,1"

python3 dlrm_data_pytorch.py --fun='extract' \
--raw-data-file=/home/datasets/recommendation/Criteo_Terabyte/day \
--extract-sample-rate=0.01 \
--extract-day="2"

echo "backup files: day_0, day_1, day_2, and rename  day_0_ext, day_1_ext, day_2_ext ..."
mv /home/datasets/recommendation/Criteo_Terabyte/day_0 /home/datasets/recommendation/Criteo_Terabyte/day_0_bk
mv /home/datasets/recommendation/Criteo_Terabyte/day_1 /home/datasets/recommendation/Criteo_Terabyte/day_1_bk
mv /home/datasets/recommendation/Criteo_Terabyte/day_2 /home/datasets/recommendation/Criteo_Terabyte/day_2_bk
mv /home/datasets/recommendation/Criteo_Terabyte/day_0_ext /home/datasets/recommendation/Criteo_Terabyte/day_0
mv /home/datasets/recommendation/Criteo_Terabyte/day_1_ext /home/datasets/recommendation/Criteo_Terabyte/day_1
mv /home/datasets/recommendation/Criteo_Terabyte/day_2_ext /home/datasets/recommendation/Criteo_Terabyte/day_2

else
    echo "day_0, day_1, day_2 have already been extracted."
fi

# Preprocess files
echo "preprocess datasets, generate bin files ..."

if [ ! -f /home/datasets/recommendation/Criteo_Terabyte/terabyte_processed_test.bin ] || [ ! -f /home/datasets/recommendation/Criteo_Terabyte/terabyte_processed_train.bin ];then
python3 dlrm_data_pytorch.py --fun='preprocess' \
--data-set=terabyte \
--raw-data-file=/home/datasets/recommendation/Criteo_Terabyte/day \
--processed-data-file=/home/datasets/recommendation/Criteo_Terabyte/terabyte_processed.npz \
--memory-map \
--data-sub-sample-rate=0.875 \
--day-num=3 \
--dataset-multiprocessing
else
    echo "bin files have already exsit"
fi

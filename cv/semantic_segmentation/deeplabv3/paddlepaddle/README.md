# DeepLab

## Model description

DeepLabv3 is a semantic segmentation architecture that improves upon DeepLabv2 with several modifications. 
To handle the problem of segmenting objects at multiple scales, modules are designed which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates. 

## Step 1: Installing

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
```

## Step 2: Download data

Download the [CityScapes Dataset](https://www.cityscapes-dataset.com/) 

```bash
# Datasets preprocessing
pip3 install cityscapesscripts

python3 tools/convert_cityscapes.py --cityscapes_path /home/datasets/cityscapes/ --num_workers 8

python3 tools/create_dataset_list.py /home/datasets/cityscapes --type cityscapes --separator ","
# CityScapes PATH as follow:
ls -al /home/datasets/cityscapes/
total 11567948
drwxr-xr-x 4 root root         227 Jul 18 03:32 .
drwxr-xr-x 6 root root         179 Jul 18 06:48 ..
-rw-r--r-- 1 root root         298 Feb 20  2016 README
drwxr-xr-x 5 root root          58 Jul 18 03:30 gtFine
-rw-r--r-- 1 root root   252567705 Jul 18 03:22 gtFine_trainvaltest.zip
drwxr-xr-x 5 root root          58 Jul 18 03:30 leftImg8bit
-rw-r--r-- 1 root root 11592327197 Jul 18 03:27 leftImg8bit_trainvaltest.zip
-rw-r--r-- 1 root root        1646 Feb 17  2016 license.txt
-rw-r--r-- 1 root root      193690 Jul 18 03:32 test.txt
-rw-r--r-- 1 root root      398780 Jul 18 03:32 train.txt
-rw-r--r-- 1 root root       65900 Jul 18 03:32 val.txt
```

## Step 3: Run DeepLab

```bash
# Make sure your dataset path is the same as above
data_dir=${data_dir:-/home/datasets/cityscapes/}
sed -i "s#: data/cityscapes#: ${data_dir}#g" configs/_base_/cityscapes.yml
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 train.py \
       --config configs/deeplabv3/deeplabv3_resnet50_os8_cityscapes_1024x512_80k.yml \
       --do_eval \
       --use_vdl
```

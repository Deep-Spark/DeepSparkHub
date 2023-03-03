# BiSeNetV2

## Model description

A novel Bilateral Segmentation Network (BiSeNet).
First design a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features.
Meanwhile, a Context Path with a fast downsampling strategy is employed to obtain sufficient receptive field.
On top of the two paths, we introduce a new Feature Fusion Module to combine features efficiently. 

## Step 1: Installing

```bash
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleSeg.git
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

## Step 3: Run BiSeNetV2

```bash
# Make sure your dataset path is the same as above
data_dir=${data_dir:-/home/datasets/cityscapes/}
sed -i "s#: data/cityscapes#: ${data_dir}#g" configs/_base_/cityscapes.yml
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
# One GPU
export CUDA_VISIBLE_DEVICES=0
python3 train.py --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml --do_eval --use_vdl --save_interval 500 --save_dir output

# Four GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 train.py \
       --config configs/bisenet/bisenet_cityscapes_1024x1024_160k.yml \
       --do_eval \
       --use_vdl
```

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 8 cards     | mIoU=73.45%                          |
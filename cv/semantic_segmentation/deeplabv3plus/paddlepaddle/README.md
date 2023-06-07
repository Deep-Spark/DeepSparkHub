# DeepLabV3+

## Model description

DeepLabv3 is a semantic segmentation architecture that improves upon DeepLabv2 with several modifications. 
To handle the problem of segmenting objects at multiple scales, modules are designed which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates. 

## Step 1: Installing

```bash
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
```

## Step 2: Download data

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure sholud look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

```bash
# Datasets preprocessing
pip3 install cityscapesscripts

python3 tools/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8

python3 tools/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","
# CityScapes PATH as follow:
ls -al /path/to/cityscapes
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

## Step 3: Run DeepLabV3+

```bash
# Change '/path/to/cityscapes' as your local Cityscapes dataset path
data_dir=/path/to/cityscapes
sed -i "s#: data/cityscapes#: ${data_dir}#g" configs/_base_/cityscapes.yml
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
# One GPU
export CUDA_VISIBLE_DEVICES=0
python3 train.py --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml --do_eval --use_vdl --save_interval 500 --save_dir output

# Four GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3 
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 train.py \
       --config configs/deeplabv3p/deeplabv3p_resnet50_os8_cityscapes_1024x512_80k.yml \
       --do_eval \
       --use_vdl
```

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 8 cards     | mIoU =80.42%                         |
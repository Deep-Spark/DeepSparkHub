# DBNet++

## Model description

Recently, segmentation-based scene text detection methods have drawn extensive attention in the scene text detection field, because of their superiority in detecting the text instances of arbitrary shapes and extreme aspect ratios, profiting from the pixel-level descriptions. However, the vast majority of the existing segmentation-based approaches are limited to their complex post-processing algorithms and the scale robustness of their segmentation models, where the post-processing algorithms are not only isolated to the model optimization but also time-consuming and the scale robustness is usually strengthened by fusing multi-scale feature maps directly. In this paper, we propose a Differentiable Binarization (DB) module that integrates the binarization process, one of the most important steps in the post-processing procedure, into a segmentation network. Optimized along with the proposed DB module, the segmentation network can produce more accurate results, which enhances the accuracy of text detection with a simple pipeline. Furthermore, an efficient Adaptive Scale Fusion (ASF) module is proposed to improve the scale robustness by fusing features of different scales adaptively. By incorporating the proposed DB and ASF with the segmentation network, our proposed scene text detector consistently achieves state-of-the-art results, in terms of both detection accuracy and speed, on five standard benchmarks.

## Step 1: Installation

```bash
# Clone PaddleOCR, branch: release/2.5
git clone -b release/2.5  https://github.com/PaddlePaddle/PaddleOCR.git

# Copy PaddleOCR 2.5 patch from toolbox
yes | cp -rf ../../../../toolbox/PaddleOCR/* PaddleOCR/
cd PaddleOCR

# install requirements.
pip3 install protobuf==3.20.3 urllib3==1.26.6
yum install -y mesa-libGL
bash ../init.sh
```

## Step 2: Preparing datasets

Download the [ICDAR2015 Dataset](https://deepai.org/dataset/icdar-2015)

```bash
# ICDAR2015 PATH as follow:
ls -al /home/datasets/ICDAR2015/text_localization
total 133420
drwxr-xr-x 4 root root      179 Jul 21 15:54 .
drwxr-xr-x 3 root root       39 Jul 21 15:50 ..
drwxr-xr-x 2 root root    12288 Jul 21 15:53 ch4_test_images
-rw-r--r-- 1 root root 44359601 Jul 21 15:51 ch4_test_images.zip
-rw-r--r-- 1 root root 90667586 Jul 21 15:51 ch4_training_images.zip
drwxr-xr-x 2 root root    24576 Jul 21 15:53 icdar_c4_train_imgs
-rw-r--r-- 1 root root   468453 Jul 21 15:54 test_icdar2015_label.txt
-rw-r--r-- 1 root root  1063118 Jul 21 15:54 train_icdar2015_label.txt

# Prepare datasets
mkdir train_data pretrain_models
ln -s /path/to/icdar2015/ train_data/icdar2015

# Pretrain
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
```

## Step 3: Training

```bash
# run training
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus $CUDA_VISIBLE_DEVICES \
    tools/train.py \
    -c configs/det/det_mv3_db.yml \
    -o Global.pretrained_model=./pretrain_models/MobileNetV3_large_x0_5_pretrained
```

## Results


| GPUs       | IPS            | ACC               |
| ------------ | ---------------- | ------------------- |
| BI-V100 x8 | 5.46 samples/s | precision: 0.9062 |

## Reference

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR.git)

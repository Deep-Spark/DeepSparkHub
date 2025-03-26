# DBNet++

## Model Description

DBNet++ is an advanced scene text detection model that combines a Differentiable Binarization (DB) module with an
Adaptive Scale Fusion (ASF) mechanism. The DB module integrates binarization directly into the segmentation network,
simplifying post-processing and improving accuracy. The ASF module enhances scale robustness by adaptively fusing
multi-scale features. This architecture enables DBNet++ to detect text of arbitrary shapes and extreme aspect ratios
efficiently, achieving state-of-the-art performance in both accuracy and speed across various text detection benchmarks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

Download [ICDAR 2015](https://deepai.org/dataset/icdar-2015) Dataset.

```bash
# ICDAR2015 PATH as follow:
$ ls -al /home/datasets/ICDAR2015/text_localization
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

### Install Dependencies

```bash
# Clone PaddleOCR, branch: release/2.5
git clone -b release/2.5  https://github.com/PaddlePaddle/PaddleOCR.git

# Copy PaddleOCR 2.5 patch from toolbox
yes | cp -rf ../../../../toolbox/PaddleOCR/* PaddleOCR/
cd PaddleOCR

# install requirements.
bash ../init.sh

# build PaddleOCR
python3 setup.py develop
```

## Model Training

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

## Model Results

| Model   | GPUs       | IPS            | ACC               |
|---------|------------|----------------|-------------------|
| DBNet++ | BI-V100 x8 | 5.46 samples/s | precision: 0.9062 |

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR.git)

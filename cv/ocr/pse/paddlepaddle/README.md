# PSE

## Model Description

PSE (Progressive Scale Expansion Network) is a deep learning model for robust text detection in natural scenes. It
addresses the challenge of detecting text with arbitrary shapes by progressively expanding text regions through a scale
expansion algorithm. PSE effectively handles complex scenarios like curved text and overlapping instances. The model's
architecture combines feature pyramid networks with a novel post-processing method, making it particularly suitable for
detecting text in diverse orientations and layouts with high accuracy.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.3.0     |  22.12  |

## Model Preparation

### Prepare Resources

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
```

### Install Dependencies

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip3 install -r requirements.txt
```

## Model Training

```bash
# Notice: modify "configs/det/det_r50_vd_pse.yml" file, set the datasets path as yours.
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/ResNet50_vd_ssld_pretrained.pdparams
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 tools/train.py -c configs/det/det_r50_vd_pse.yml -o Global.use_amp=True Global.scale_loss=1024.0 Global.use_dynamic_loss_scaling=True
```

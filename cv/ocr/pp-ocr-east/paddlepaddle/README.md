# PP-OCR-EAST
## Model description

EAST (Efficient and Accurate Scene Text Detector) is a deep learning model designed for detecting and recognizing text in natural scene images. 
It was developed by researchers at the School of Electronic Information and Electrical Engineering, Shanghai Jiao Tong University, and was presented in a research paper in 2017.

## Step 1: Installation
```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip3 install -r requirements.txt
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

```

## Step 3: Training

```bash
# Notice: modify "configs/det/det_mv3_east.yml" file, set the datasets path as yours.
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/det_mv3_east.yml -o Global.use_visualdl=True 
```

## Results

GPUs|FPS|ACC
----|---|---
BI-V100 x8|50.08|hmean:0.7711, precision: 0.7752

## Reference
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR.git)

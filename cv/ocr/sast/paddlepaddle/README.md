# SAST

## Model Description

SAST is a cutting-edge segmentation-based text detector designed for recognizing scene text of arbitrary shapes.
Leveraging a context attended multi-task learning framework anchored on a Fully Convolutional Network (FCN), it adeptly
learns geometric properties to reconstruct text regions into polygonal shapes. Incorporating a Context Attention Block,
SAST captures long-range pixel dependencies for improved segmentation accuracy, while its Point-to-Quad assignment
method efficiently clusters pixels into text instances by merging high-level and low-level information. Demonstrated to
be highly effective across several benchmarks like ICDAR2015 and SCUT-CTW1500, SAST not only shows superior accuracy but
also operates efficiently, achieving significant performance metrics such as running at 27.63 FPS on a NVIDIA Titan Xp
with a high detection accuracy, making it a notable solution for arbitrary-shaped text detection challenges.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.1     |  24.03  |

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
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/
pip3 install -r requirements.txt
```

## Model Training

```bash
# Notice: modify "configs/det/det_r50_vd_sast_icdar15.yml" file, set the datasets path as yours.
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/ResNet50_vd_ssld_pretrained.pdparams
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7' tools/train.py -c configs/det/det_r50_vd_sast_icdar15.yml -o Global.use_visualdl=True \
>train.log 2>&1 &
```

## Model Results

| Model | GPU        | FPS                     | ACC                      |
|-------|------------|-------------------------|--------------------------|
| SAST  | BI-V100 x8 | ips: 11.24631 samples/s | hmean: 0.817155756207675 |

## References

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR.git)

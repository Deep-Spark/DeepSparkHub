# Mask R-CNN

## Model Description

Mask R-CNN is an advanced instance segmentation model that extends Faster R-CNN by adding a parallel branch for
predicting object masks. It efficiently detects objects in an image while simultaneously generating high-quality
segmentation masks for each instance. Mask R-CNN maintains the two-stage architecture of Faster R-CNN but introduces a
fully convolutional network for mask prediction. This model achieves state-of-the-art performance on tasks like object
detection, instance segmentation, and human pose estimation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.3.0     |  22.12  |

## Model Preparation

### Prepare Resources

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone --recursive https://github.com/PaddlePaddle/PaddleDetection.git -b release2.6 --depth=1
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection/
# Get COCO Dataset
python3 dataset/coco/download_coco.py
or
ln -s /path/to/coco2017 dataset/coco
```

### Install Dependencies

```bash
pip install -r requirements.txt
python3 setup.py install
```

## Model Training

```bash
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml --use_vdl=true --eval
```

## Model Results

| Model      | GPU        | FP32                          |
|------------|------------|-------------------------------|
| Mask R-CNN | BI-V100 x8 | bbox=38.8,FPS=7.5,BatchSize=1 |

## Reference

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
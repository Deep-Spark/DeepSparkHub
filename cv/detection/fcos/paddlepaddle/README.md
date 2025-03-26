# FCOS

## Model Description

FCOS (Fully Convolutional One-Stage Object Detection) is an anchor-free object detection model that predicts bounding
boxes directly without anchor boxes. It uses a fully convolutional network to detect objects by predicting per-pixel
bounding boxes and class labels. FCOS simplifies the detection pipeline, reduces hyperparameters, and achieves
competitive performance on benchmarks like COCO. Its center-ness branch helps suppress low-quality predictions, making
it efficient and effective for various detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/PaddlePaddle/PaddleDetection.git -b release2.6 --depth=1

cd PaddleDetection/
# Get COCO Dataset
python3 dataset/coco/download_coco.py
```

### Install Dependencies

```bash
pip install -r requirements.txt
python3 setup.py install
```

## Model Training

```bash
# Multi-GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/fcos/fcos_r50_fpn_1x_coco.yml --eval

# Single-GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/fcos/fcos_r50_fpn_1x_coco.yml --eval

# Note: The default learning rate is optimized for multi-GPU training (8x GPU). If using single GPU training,
# you need to adjust the learning rate in the config accordingly (e.g., divide by 8).
```

## Model Results

 | Model | GPU        | FPS  | Train Epochs | Box AP |
 |-------|------------|------|--------------|--------|
 | FCOS  | BI-V100 x8 | 8.24 | 12           | 39.7   |

## Reference

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
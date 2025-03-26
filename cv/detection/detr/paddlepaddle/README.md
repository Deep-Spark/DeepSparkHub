# DETR

## Model Description

DETR (DEtection TRansformer) is a novel object detection model that replaces traditional convolutional methods with a
transformer-based architecture. It treats object detection as a direct set prediction problem, eliminating the need for
anchor boxes and non-maximum suppression. DETR uses a transformer encoder-decoder structure to process image features
and predict object bounding boxes and classes simultaneously. This end-to-end approach simplifies the detection pipeline
while achieving competitive performance on benchmarks like COCO, offering a new paradigm for object detection tasks.

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
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/detr/detr_r50_1x_coco.yml --eval

# Single-GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/detr/detr_r50_1x_coco.yml --eval

# Finetune
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/detr/detr_r50_1x_coco.yml -o pretrain_weights=https://paddledet.bj.bcebos.com/models/detr_r50_1x_coco.pdparams --eval

# Note: The default learning rate is optimized for multi-GPU training (8x GPU). If using single GPU training,
# you need to adjust the learning rate in the config accordingly (e.g., divide by 8).

```

## Model Results

| Model | GPU        | learning rate | FPS   | Train Epochs | Box AP |
|-------|------------|---------------|-------|--------------|--------|
| DETR  | BI-V100 x8 | 0.00001       | 14.64 | 1            | 42.0   |

## Reference

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
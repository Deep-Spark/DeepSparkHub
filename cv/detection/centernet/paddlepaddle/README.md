# CenterNet

## Model Description

CenterNet is an efficient object detection model that represents objects as single points (their bounding box centers)
rather than traditional bounding boxes. It uses keypoint estimation to locate centers and regresses other object
properties like size and orientation. This approach eliminates the need for anchor boxes and non-maximum suppression,
making it simpler and faster. CenterNet achieves state-of-the-art speed-accuracy trade-offs on benchmarks like COCO and
can be extended to 3D detection and pose estimation tasks.

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
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml --eval

# Single-GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml --eval

# Finetune
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml -o pretrain_weights=https://bj.bcebos.com/v1/paddledet/models/centernet_r50_140e_coco.pdparams --eval

# Note: The default learning rate is optimized for multi-GPU training (8x GPU). If using single GPU training,
# you need to adjust the learning rate in the config accordingly (e.g., divide by 8).

```

## Model Results

| GPU        | learning rate | FPS   | Train Epochs | mAP  |
|------------|---------------|-------|--------------|------|
| BI-V100 x8 | 0.00005       | 10.85 | 3            | 38.5 |

## Reference
[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
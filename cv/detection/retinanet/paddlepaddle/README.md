# RetinaNet

## Model Description

RetinaNet is a state-of-the-art object detection model that addresses the class imbalance problem in dense detection
through its novel Focal Loss. It uses a Feature Pyramid Network (FPN) backbone to detect objects at multiple scales
efficiently. RetinaNet achieves high accuracy while maintaining competitive speed, making it suitable for various
detection tasks. Its single-stage architecture combines the accuracy of two-stage detectors with the speed of
single-stage approaches, offering an excellent balance between performance and efficiency.

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
# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval

# 1 GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval

## Hint：Default LR is for "8x GPU", modify it if you're using single card for training (e.g. divide by 8).
```

## Model Results

| Model     | GPU        | FPS  | Train Epochs | Box AP |
|-----------|------------|------|--------------|--------|
| RetinaNet | BI-V100 x8 | 6.58 | 12           | 37.3   |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
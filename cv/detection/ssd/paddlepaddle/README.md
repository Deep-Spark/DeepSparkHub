# SSD

## Model Description

SSD (Single Shot MultiBox Detector) is a fast and efficient object detection model that predicts bounding boxes and
class scores in a single forward pass. It uses a set of default boxes at different scales and aspect ratios across
multiple feature maps to detect objects of various sizes. SSD combines predictions from different layers to handle
objects at different resolutions, offering a good balance between speed and accuracy for real-time detection tasks.

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

git clone https://github.com/PaddlePaddle/PaddleDetection.git -b release2.6 --depth=1

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

Notice: modify configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml file, modify the datasets path as yours.

```bash
# Train
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1
python3 -u -m paddle.distributed.launch --gpus 0,1 tools/train.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --eval
```

## Model Results

| Model | GPU        | FP32                              |
|-------|------------|-----------------------------------|
| SSD   | BI-V100 x2 | bbox=73.62,FPS=45.49,BatchSize=32 |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

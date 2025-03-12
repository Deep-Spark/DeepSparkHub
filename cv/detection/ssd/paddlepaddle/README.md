# SSD

## Model Description

SSD (Single Shot MultiBox Detector) is a fast and efficient object detection model that predicts bounding boxes and
class scores in a single forward pass. It uses a set of default boxes at different scales and aspect ratios across
multiple feature maps to detect objects of various sizes. SSD combines predictions from different layers to handle
objects at different resolutions, offering a good balance between speed and accuracy for real-time detection tasks.

## Model Preparation

### Prepare Resources

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git

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

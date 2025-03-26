# YOLOv3

## Model Description

YOLOv3 is a real-time object detection model that builds upon its predecessors with improved accuracy while maintaining
speed. It uses a deeper backbone network and multi-scale predictions to detect objects of various sizes. YOLOv3 achieves
competitive performance with faster inference times compared to other detectors. It processes images in a single forward
pass, making it efficient for real-time applications. The model balances speed and accuracy, making it popular for
practical detection tasks.

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
# Make sure your dataset path is the same as above.
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
    tools/train.py \
    -c configs/yolov3/yolov3_darknet53_270e_coco.yml \
    -o TrainReader.batch_size=16 LearningRate.base_lr=0.002 worker_num=4 \
    --use_vdl=true \
    --eval
```

## Reference

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
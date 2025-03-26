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
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── train2017.txt
├── val2017.txt
└── ...
```

```bash
mkdir -p ./datasets/
ln -s /path/to/coco2017 ./datasets/coco
```

Download from <https://download.pytorch.org/models/resnet50-0676ba61.pth> and mv to /root/.cache/torch/hub/checkpoints/.

```bash
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
mkdir -p /root/.cache/torch/hub/checkpoints/
mv resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install -r requirements.txt
```

## Model Training

```bash
# Single Card
python3 train.py --data-path ./datasets/coco --dataset coco --model maskrcnn_resnet50_fpn --lr 0.001 --batch-size 4

# AMP
python3 train.py --data-path ./datasets/coco --dataset coco --model maskrcnn_resnet50_fpn --lr 0.001 --batch-size 1 --amp

# DDP
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --data-path ./datasets/coco --dataset coco --model maskrcnn_resnet50_fpn --wd 0.000001 --lr 0.001 --batch-size 4
```

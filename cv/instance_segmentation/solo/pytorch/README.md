# SOLO

## Model Description

SOLO (Segmenting Objects by Locations) is an innovative instance segmentation model that simplifies the task by
introducing "instance categories". It converts instance segmentation into a classification problem by assigning
categories to each pixel based on an object's location and size. Unlike traditional methods, SOLO directly predicts
instance masks without complex post-processing or region proposals. This approach achieves competitive accuracy with
Mask R-CNN while offering a simpler and more flexible framework for instance-level recognition tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

```bash
mkdir -p data/coco
cd data/coco
```

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

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

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 --depth=1
cd mmdetection
pip install -v -e .

# Prepare resnet50-0676ba61.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
```

## Model Training

```bash
# One single GPU
python3 tools/train.py configs/solo/solo_r50_fpn_1x_coco.py

# Multiple GPUs on one machine
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/solo/solo_r50_fpn_1x_coco.py 8
```

## Model Results

| Model | GPU        | mAP(0.5:0.95) |
|-------|------------|---------------|
| SOLO  | BI-V100 x8 | 0.361         |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection)

# ATSS

## Model Description

ATSS (Adaptive Training Sample Selection) is an innovative object detection framework that bridges the gap between
anchor-based and anchor-free methods. It introduces an adaptive mechanism to select positive and negative training
samples based on statistical characteristics of objects, improving detection accuracy. ATSS automatically determines
optimal sample selection thresholds, eliminating the need for manual tuning. This approach enhances both anchor-based
and anchor-free detectors, achieving state-of-the-art performance on benchmarks like COCO without additional
computational overhead.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
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

ATSS model is using MMDetection toolbox. Before you run this model, you need to setup MMDetection first.

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
```

## Model Training

```bash
# Make soft link to dataset
cd mmdetection/
mkdir -p data/
ln -s /path/to/coco2017 data/coco

# Prepare resnet50-0676ba61.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

# On single GPU
python3 tools/train.py configs/atss/atss_r50_fpn_1x_coco.py

sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py 8
```

## Model Results

| GPU        | FP32     |
|------------|----------|
| BI-V100 x8 | MAP=39.5 |

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection)

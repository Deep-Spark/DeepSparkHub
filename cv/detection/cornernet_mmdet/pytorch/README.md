# CornerNet

## Model Description

CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the
top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired
keypoints, we eliminate the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In
addition to our novel formulation, we introduce corner pooling, a new type of pooling layer that helps the network
better localize corners. Experiments show that CornerNet achieves a 42.2% AP on MS COCO, outperforming all existing
one-stage detectors.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

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

CornerNet model is using MMDetection toolbox. Before you run this model, you need to setup MMDetection first.

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

# On single GPU
python3 tools/train.py configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py

sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/cornernet/cornernet_hourglass104_8xb6-210e-mstest_coco.py 8
```

## Model Results

| Model     | GPU        | FP32     |
|-----------|------------|----------|
| CornerNet | BI-V100 x8 | MAP=41.2 |

## Reference
[mmdetection](https://github.com/open-mmlab/mmdetection)

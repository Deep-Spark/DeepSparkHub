# RTMDet

> [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/pdf/2212.07784v2.pdf)

<!-- [ALGORITHM] -->

## Model description

In this paper, we aim to design an efficient real-time object detector that exceeds the YOLO series and is easily extensible for many object recognition tasks such as instance segmentation and rotated object detection. To obtain a more efficient model architecture, we explore an architecture that has compatible capacities in the backbone and neck, constructed by a basic building block that consists of large-kernel depth-wise convolutions. We further introduce soft labels when calculating matching costs in the dynamic label assignment to improve accuracy. Together with better training techniques, the resulting object detector, named RTMDet, achieves 52.8% AP on COCO with 300+ FPS on an NVIDIA 3090 GPU, outperforming the current mainstream industrial detectors. RTMDet achieves the best parameter-accuracy trade-off with tiny/small/medium/large/extra-large model sizes for various application scenarios and obtains new state-of-the-art performance on real-time instance segmentation and rotated object detection. We hope the experimental results can provide new insights into designing versatile real-time object detectors for many object recognition tasks.

## Step 1: Installation

RTMDet model uses the MMDetection toolbox. Before you run this model, you need to set up MMDetection first.

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

## Step 2: Preparing datasets

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

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
## Step 3: Training

```bash
# Make soft link to dataset
cd mmdetection/
mkdir -p data/
ln -s /path/to/coco2017 data/coco

# Prepare cspnext-tiny_imagenet_600e.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget -O /root/.cache/torch/hub/checkpoints/cspnext-tiny_imagenet_600e.pth https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth

# On single GPU
python3 tools/train.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py

sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py 8
```

## Results

|GPUs|FPS|box AP|
|:---:|:---:|:---:|
|BI-V100|172.5|0.4090|

## Reference
- [mmdetection](https://github.com/open-mmlab/mmdetection)
# RTMDet

> [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/pdf/2212.07784v2.pdf)

<!-- [ALGORITHM] -->

## Abstract

In this paper, we aim to design an efficient real-time object detector that exceeds the YOLO series and is easily extensible for many object recognition tasks such as instance segmentation and rotated object detection. To obtain a more efficient model architecture, we explore an architecture that has compatible capacities in the backbone and neck, constructed by a basic building block that consists of large-kernel depth-wise convolutions. We further introduce soft labels when calculating matching costs in the dynamic label assignment to improve accuracy. Together with better training techniques, the resulting object detector, named RTMDet, achieves 52.8% AP on COCO with 300+ FPS on an NVIDIA 3090 GPU, outperforming the current mainstream industrial detectors. RTMDet achieves the best parameter-accuracy trade-off with tiny/small/medium/large/extra-large model sizes for various application scenarios, and obtains new state-of-the-art performance on real-time instance segmentation and rotated object detection. We hope the experimental results can provide new insights into designing versatile real-time object detectors for many object recognition tasks.

## Step 1: Installation

RTMDet model is using MMDetection toolbox. Before you run this model, you need to setup MMDetection first.

```bash
# Install mmcv
pushd ../../../../toolbox/MMDetection/patch/mmcv/v2.0.0rc4/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
popd

# Install mmdetection
pushd ../../../../toolbox/MMDetection/
git clone --depth 1 -b v2.22.0 https://github.com/open-mmlab/mmdetection.git
cp -r -T patch/mmdetection/ mmdetection/

cd mmdetection/
bash clean_mmdetection.sh
bash build_mmdetection.sh

pip3 install build_pip/mmdet-2.22.0+corex*-py3-none-any.whl
popd

# Install libGL
yum install -y mesa-libGL

# Install urllib3
pip3 install urllib3==1.26.6

cd ../../../../toolbox/MMDetection/mmdetection
pip3 install -v -e .
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

# On single GPU
python3 tools/train.py configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py 8
```
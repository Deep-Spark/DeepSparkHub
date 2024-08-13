# YOLOv10

## Model description

YOLOv10, built on the Ultralytics Python package by researchers at Tsinghua University, introduces a new approach to real-time object detection, addressing both the post-processing and model architecture deficiencies found in previous YOLO versions. By eliminating non-maximum suppression (NMS) and optimizing various model components, YOLOv10 achieves state-of-the-art performance with significantly reduced computational overhead. Extensive experiments demonstrate its superior accuracy-latency trade-offs across multiple model scales.

## Step 1: Installation

```bash
# CentOS
yum install -y mesa-libGL
# Ubuntu
apt install -y libgl1-mesa-glx
```

## Step 2: Preparing datasets

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

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
# make soft link to coco dataset
mkdir -p datasets/
ln -s /PATH/TO/COCO ./datasets/coco
```

## Step 3: Training

```bash
# get yolov10 code
git clone https://github.com/THU-MIG/yolov10.git
cd yolov10
sed -i 's/^torch/# torch/g' requirements.txt
pip install -r requirements.txt
```

### Multiple GPU training

```bash
yolo detect train data=coco.yaml model=yolov10n.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
```

## Reference

[YOLOv10](https://github.com/THU-MIG/yolov10)

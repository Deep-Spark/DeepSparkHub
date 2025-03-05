# YOLOF(You Only Look One-level Feature)

## Model description

This paper revisits feature pyramids networks (FPN) for one-stage detectors and points out that the success of FPN is due to its divide-and-conquer solution to the optimization problem in object detection rather than multi-scale feature fusion. From the perspective of optimization, we introduce an alternative way to address the problem instead of adopting the complex feature pyramids - {\em utilizing only one-level feature for detection}. Based on the simple and efficient solution, we present You Only Look One-level Feature (YOLOF). In our method, two key components, Dilated Encoder and Uniform Matching, are proposed and bring considerable improvements. Extensive experiments on the COCO benchmark prove the effectiveness of the proposed model. Our YOLOF achieves comparable results with its feature pyramids counterpart RetinaNet while being 2.5x faster. Without transformer layers, YOLOF can match the performance of DETR in a single-level feature manner with 7x less training epochs. With an image size of 608x608, YOLOF achieves 44.3 mAP running at 60 fps on 2080Ti, which is 13% faster than YOLOv4. Code is available at \url{https://github.com/megvii-model/YOLOF}.

## Step 1: Installing packages

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
mkdir -p data
ln -s /path/to/coco2017 data/coco

# Prepare resnet50_caffe-788b5fa3.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget -O /root/.cache/torch/hub/checkpoints/resnet50_caffe-788b5fa3.pth https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth
```

## Step 3: Training

#### Training on a single GPU

```bash
python3 tools/train.py configs/yolof/yolof_r50-c5_8xb8-1x_coco.py
```

#### Training on multiple GPUs

```bash
sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/yolof/yolof_r50-c5_8xb8-1x_coco.py 8
```

## Reference

- [mmdetection](https://github.com/open-mmlab/mmdetection)
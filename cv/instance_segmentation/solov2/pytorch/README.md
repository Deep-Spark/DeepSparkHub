# SOLOV2

## Model description

In this work, we aim at building a simple, direct, and fast instance segmentation framework with strong performance. We follow the principle of the SOLO method of Wang et al. "SOLO: segmenting objects by locations". Importantly, we take one step further by dynamically learning the mask head of the object segmenter such that the mask head is conditioned on the location. Specifically, the mask branch is decoupled into a mask kernel branch and mask feature branch, which are responsible for learning the convolution kernel and the convolved features respectively. Moreover, we propose Matrix NMS (non maximum suppression) to significantly reduce the inference time overhead due to NMS of masks. Our Matrix NMS performs NMS with parallel matrix operations in one shot, and yields better results. We demonstrate a simple direct instance segmentation system, outperforming a few state-of-the-art methods in both speed and accuracy. A light-weight version of SOLOv2 executes at 31.3 FPS and yields 37.1% AP. Moreover, our state-of-the-art results in object detection (from our mask byproduct) and panoptic segmentation show the potential to serve as a new strong baseline for many instance-level recognition tasks besides instance segmentation.

## Step 1: Installation

```bash
# Install mmcv
pushd ../../../../toolbox/MMDetection
bash prepare_mmcv.sh v2.0.0rc4
popd

# Install mmdetection
git clone -b v3.2.0 https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip3 install -r requirements.txt
python3 setup.py develop

# Install mmengine
pip3 install mmengine==0.8.3

# Prepare resnet50-0676ba61.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

# Install others
pip3 install yapf==0.31.0 urllib3==1.26.18
yum install -y mesa-libGL
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

sed -i 's/python /python3 /g' tools/dist_train.sh

# On single GPU
python3 tools/train.py configs/solov2/solov2_r50_fpn_1x_coco.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/solov2/solov2_r50_fpn_1x_coco.py 8
```

## Results

|    GPUs    | FPS |
| ---------- | --------- |
| BI-V100 x8 | 21.26 images/s |

## Reference

- [mmdetection](https://github.com/open-mmlab/mmdetection/tree/v3.2.0/configs/solov2)

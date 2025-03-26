# RTMDet

## Model Description

RTMDet is a highly efficient real-time object detection model designed to surpass YOLO series performance. It features a
balanced architecture with large-kernel depth-wise convolutions and dynamic label assignment using soft labels. RTMDet
achieves state-of-the-art accuracy with exceptional speed, reaching 300+ FPS on modern GPUs. The model offers various
sizes for different applications and excels in tasks like instance segmentation and rotated object detection. Its design
provides insights for versatile real-time detection systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

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

### Install Dependencies

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

## Model Training

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

## Model Results

| Model  | GPU     | FPS   | box AP |
|--------|---------|-------|--------|
| RTMDet | BI-V100 | 172.5 | 0.4090 |

## References

- [Paper](https://arxiv.org/pdf/2212.07784v2.pdf)
- [mmdetection](https://github.com/open-mmlab/mmdetection)

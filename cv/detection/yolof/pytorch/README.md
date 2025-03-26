# YOLOF

## Model Description

YOLOF (You Only Look One-level Feature) is an efficient object detection model that challenges the necessity of feature
pyramids. It demonstrates that using a single-level feature with proper optimization can achieve comparable results to
multi-level approaches. YOLOF introduces two key components: Dilated Encoder for capturing multi-scale context and
Uniform Matching for balanced positive samples. The model achieves competitive accuracy with RetinaNet while being 2.5x
faster, making it suitable for real-time detection tasks.

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
mkdir -p data
ln -s /path/to/coco2017 data/coco
```

Prepare resnet50_caffe-788b5fa3.pth, skip this if fast network

```bash
mkdir -p /root/.cache/torch/hub/checkpoints/
wget -O /root/.cache/torch/hub/checkpoints/resnet50_caffe-788b5fa3.pth https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth
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
```

## Model Training

```bash
# Training on a single GPU
python3 tools/train.py configs/yolof/yolof_r50-c5_8xb8-1x_coco.py

# Training on multiple GPUs
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/yolof/yolof_r50-c5_8xb8-1x_coco.py 8
```

## References

- [mmdetection](https://github.com/open-mmlab/mmdetection)

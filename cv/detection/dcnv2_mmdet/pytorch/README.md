# DCNv2

## Model Description

DCNv2 (Deformable Convolutional Networks v2) is an advanced convolutional neural network architecture that enhances
spatial transformation capabilities. It improves upon DCN by introducing a modulation mechanism for more precise
deformation modeling and better focus on relevant image regions. DCNv2 integrates deformable convolution more
comprehensively throughout the network, enabling superior adaptation to object geometry. This results in
state-of-the-art performance on object detection and instance segmentation tasks, particularly on benchmarks like COCO,
while maintaining computational efficiency.

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

DCNv2 model is using MMDetection toolbox. Before you run this model, you need to setup MMDetection first.

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
python3 tools/train.py configs/dcnv2/faster-rcnn_r50-mdconv-c3-c5_fpn_1x_coco.py

sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/dcnv2/faster-rcnn_r50-mdconv-c3-c5_fpn_1x_coco.py 8
```

## Model Results

 | GPU   | FP32       | mAP  |
 |-------|------------|------|
 | DCNv2 | BI-V100 x8 | 41.2 |

## Reference
[mmdetection](https://github.com/open-mmlab/mmdetection)

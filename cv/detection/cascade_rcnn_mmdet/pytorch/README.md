# Cascade R-CNN

## Model Description

Cascade R-CNN is a multi-stage object detection framework that progressively improves detection quality through a
sequence of detectors trained with increasing IoU thresholds. Each stage refines the bounding boxes from the previous
stage, addressing the paradox of high-quality detection by minimizing overfitting and ensuring quality consistency
between training and inference. This architecture achieves state-of-the-art performance on various datasets, including
COCO, and can be extended to instance segmentation tasks, outperforming models like Mask R-CNN.

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

Cascade R-CNN model is using MMDetection toolbox. Before you run this model, you need to setup MMDetection first.

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
python3 tools/train.py  configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py 

sed -i 's/python /python3 /g' tools/dist_train.sh

# Multiple GPUs on one machine
bash tools/dist_train.sh  configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py  8
```

## Model Results

| GPU        | FP32     |
|------------|----------|
| BI-V100 x8 | MAP=40.4 |

## References

- [Paper](https://arxiv.org/abs/1906.09756)

# DBNet++

## Model description

Recently, segmentation-based scene text detection methods have drawn extensive attention in the scene text detection field, because of their superiority in detecting the text instances of arbitrary shapes and extreme aspect ratios, profiting from the pixel-level descriptions. However, the vast majority of the existing segmentation-based approaches are limited to their complex post-processing algorithms and the scale robustness of their segmentation models, where the post-processing algorithms are not only isolated to the model optimization but also time-consuming and the scale robustness is usually strengthened by fusing multi-scale feature maps directly. In this paper, we propose a Differentiable Binarization (DB) module that integrates the binarization process, one of the most important steps in the post-processing procedure, into a segmentation network. Optimized along with the proposed DB module, the segmentation network can produce more accurate results, which enhances the accuracy of text detection with a simple pipeline. Furthermore, an efficient Adaptive Scale Fusion (ASF) module is proposed to improve the scale robustness by fusing features of different scales adaptively. By incorporating the proposed DB and ASF with the segmentation network, our proposed scene text detector consistently achieves state-of-the-art results, in terms of both detection accuracy and speed, on five standard benchmarks.

## Step 1: Installation

```bash
# Install mmcv
pushd ../../../../toolbox/MMDetection
bash prepare_mmcv.sh v2.0.0rc4
popd

# Install  mmdet and mmocr
pip3 install mmdet==3.1.0

git clone -b v1.0.1 https://github.com/open-mmlab/mmocr.git
cd mmocr
pip3 install -r requirements.txt
python3 setup.py develop

# Install mmengine
pip3 install mmengine==0.8.3
yum install -y mesa-libGL

# Prepare resnet50-0676ba61.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
```

## Step 2: Preparing datasets

```bash
mkdir data
python3 tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```

## Step 3: Training

```bash
sed -i 's/val_interval=20/val_interval=1200/g' configs/textdet/_base_/schedules/schedule_sgd_1200e.py
sed -i 's/python /python3 /g' tools/dist_train.sh

# On single GPU
python3 tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015.py 8
```
## Results

|    GPUs    | Precision | Recall | Hmean |
| ---------- | --------- | ------ | ----- |
| BI-V100 x8 | 0.8823 | 0.8156 | 0.8476 |

## Reference

- [mmocr](https://github.com/open-mmlab/mmocr/tree/v1.0.1/configs/textdet/dbnetpp)

# DBNet++

## Model Description

DBNet++ is an advanced scene text detection model that combines a Differentiable Binarization (DB) module with an
Adaptive Scale Fusion (ASF) mechanism. The DB module integrates binarization directly into the segmentation network,
simplifying post-processing and improving accuracy. The ASF module enhances scale robustness by adaptively fusing
multi-scale features. This architecture enables DBNet++ to detect text of arbitrary shapes and extreme aspect ratios
efficiently, achieving state-of-the-art performance in both accuracy and speed across various text detection benchmarks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

```bash
mkdir data
python3 tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# Install mmdet and mmocr
pip3 install mmdet==3.3.0

git clone -b v1.0.1 https://github.com/open-mmlab/mmocr.git
cd mmocr
pip3 install -r requirements.txt
python3 setup.py develop

# Prepare resnet50-0676ba61.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
```

## Model Training

```bash
sed -i 's/val_interval=20/val_interval=1200/g' configs/textdet/_base_/schedules/schedule_sgd_1200e.py
sed -i 's/python /python3 /g' tools/dist_train.sh
# match mmdet 3.3.0
sed -i 's/3.2.0/3.4.0/g' mmocr/__init__.py

# On single GPU
python3 tools/train.py configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/textdet/dbnetpp/dbnetpp_resnet50_fpnc_1200e_icdar2015.py 8
```

## Model Results

| Model   | GPU        | Precision | Recall | Hmean  |
|---------|------------|-----------|--------|--------|
| DBNet++ | BI-V100 x8 | 0.8823    | 0.8156 | 0.8476 |

## References

- [mmocr](https://github.com/open-mmlab/mmocr/tree/v1.0.1/configs/textdet/dbnetpp)

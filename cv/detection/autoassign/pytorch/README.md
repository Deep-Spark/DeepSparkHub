# AutoAssign

## Model Description

AutoAssign is an anchor-free object detection model that introduces a fully differentiable label assignment mechanism.
It combines Center Weighting and Confidence Weighting to adaptively determine positive and negative samples during
training. Center Weighting adjusts category-specific prior distributions, while Confidence Weighting customizes
assignment strategies for each instance. This approach eliminates the need for manual anchor design and achieves
appearance-aware detection through automatic sample selection, resulting in improved performance and reduced human
intervention in the detection process.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```bash
mkdir -p data 
ln -s /path/to/coco/ ./data

# Prepare resnet50_msra-5891d200.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.openmmlab.com/pretrain/third_party/resnet50_msra-5891d200.pth -O /root/.cache/torch/hub/checkpoints/resnet50_msra-5891d200.pth
```

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
# One single GPU
python3 tools/train.py configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py

# Multiple GPUs on one machine
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/autoassign/autoassign_r50-caffe_fpn_1x_coco.py 8
```

## References

[mmdetection](https://github.com/open-mmlab/mmdetection)

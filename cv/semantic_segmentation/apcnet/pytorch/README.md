# APCNet

## Model Description

Adaptive Pyramid Context Network (APCNet) for semantic segmentation. APCNet adaptively constructs multi-scale contextual
representations with multiple well-designed Adaptive Context Modules (ACMs). Specifically, each ACM leverages a global
image representation as a guidance to estimate the local affinity coefficients for each sub-region. And then calculates
a context vector with these affinities.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the
Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure
sholud look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

```shell
mkdir -p data/
ln -s /path/to/cityscapes data/cityscapes
```

### Install Dependencies

```shell
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# install mmsegmentation
git clone -b v1.2.2 https://github.com/open-mmlab/mmsegmentation.git --depth=1
cd mmsegmentation/
pip install -v -e .

pip install ftfy
```

## Model Training

```shell
# Training on single card
python3 tools/train.py configs/apcnet/apcnet_r50-d8_4xb2-80k_cityscapes-512x1024.py

# Training on mutil-cards
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/apcnet/apcnet_r50-d8_4xb2-80k_cityscapes-512x1024.py 8
```

## Model Results

| Model  | Backbone | Crop Size | Lr schd | Mem (GB) | mIoU (BI x 4) |
|--------|----------|-----------|---------|----------|---------------|
| APCNet | R-50-D8  | 512x1024  | 40000   | 7.7      | 77.53         |

## References

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

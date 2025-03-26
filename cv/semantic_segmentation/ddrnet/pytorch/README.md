# DDRNet

## Model Description

we proposed a family of efficient backbones specially designed for real-time semantic segmentation. The proposed deep
dual-resolution networks (DDRNets) are composed of two deep branches between which multiple bilateral fusions are
performed. Additionally, we design a new contextual information extractor named Deep Aggregation Pyramid Pooling Module
(DAPPM) to enlarge effective receptive fields and fuse multi-scale context based on low-resolution feature maps. Our
method achieves a new state-of-the-art trade-off between accuracy and speed on both Cityscapes and CamVid dataset.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the
Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure
should look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

```bash
mkdir -p data/
ln -s /path/to/cityscapes data/
```

### Install Dependencies

```bash
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
python3 tools/train.py configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py

# Training on mutil-cards
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/ddrnet/ddrnet_23-slim_in1k-pre_2xb6-120k_cityscapes-1024x1024.py 8
```

## Model Results

| GPU        | Crop Size | Lr schd | FPS    | mIoU |
|------------|-----------|--------:|--------|-----:|
| BI-V100 x8 | 1024x1024 |   12000 | 33.085 | 74.8 |

## References

- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

# TTSR

## Model Description

TTSR (Texture Transformer Network for Image Super-Resolution) is an innovative deep learning model that enhances image
super-resolution using reference images. It employs a transformer architecture where low-resolution and reference images
are formulated as queries and keys. TTSR features four key modules: texture extractor, relevance embedding,
hard-attention for texture transfer, and soft-attention for texture synthesis. This design enables effective texture
transfer through attention mechanisms, allowing for high-quality image reconstruction at various magnification levels
(1x to 4x).

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```bash
mkdir -p data/
cd data
# Download CUFED Dataset from [homepage](https://zzutk.github.io/SRNTT-Project-Page)
# the folder would be like:
data/CUFED/
└── input
├── ref
└── CUFED5

# Prepare vgg19-dcbb9e9d.pth, skip this if fast network
mkdir -p /root/.cache/torch/hub/checkpoints/
wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth -O /root/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/open-mmlab/mmagic.git -b v1.2.0 --depth=1
cd mmagic/
pip3 install -e . -v

sed -i 's/diffusers.models.unet_2d_condition/diffusers.models.unets.unet_2d_condition/g' mmagic/models/editors/vico/vico_utils.py
pip install albumentations
```

## Model Training

```shell
# Training on single card
python3 tools/train.py configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py

# Mutiple GPUs on one machine
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/ttsr/ttsr-gan_x4c64b16_1xb9-500k_CUFED.py 8
```

## References

- [mmagic](https://github.com/open-mmlab/mmagic)
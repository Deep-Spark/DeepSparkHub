# LIIF

## Model Description

LIIF (Local Implicit Image Function) is an innovative deep learning model for continuous image representation. It uses
neural networks to learn a function that can predict RGB values at any continuous coordinate within an image, enabling
arbitrary resolution representation. LIIF combines 2D deep features with coordinate inputs to generate high-quality
images, even at resolutions 30x higher than training data. This approach bridges discrete and continuous image
representations, outperforming traditional resizing methods and supporting tasks with varying image sizes.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```shell
# Download DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/ or you can follow this tools/dataset_converters/div2k/README.md
mkdir -p data/DIV2K
```

### Install Dependencies

```shell
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
# One single GPU
python3 tools/train.py configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py

# Mutiple GPUs on one machine
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/liif/liif-edsr-norm_c64b16_1xb16-1000k_div2k.py 8
```

## Model Results

| GPUs       | FP16  | FPS | PSNR  |
|------------|-------|-----|-------|
| BI-V100 x8 | False | 684 | 26.87 |

## References

- [mmagic](https://github.com/open-mmlab/mmagic)

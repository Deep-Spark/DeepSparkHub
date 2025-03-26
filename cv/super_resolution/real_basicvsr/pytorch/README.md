# RealBasicVSR

## Model Description

RealBasicVSR is an advanced video super-resolution model designed for real-world applications. It addresses challenges
in handling diverse and complex degradations through a novel image pre-cleaning module that balances detail synthesis
and artifact suppression. The model introduces a stochastic degradation scheme to reduce training time while maintaining
performance, and emphasizes the use of longer sequences over larger batches for more effective temporal information
utilization. RealBasicVSR demonstrates superior quality and efficiency in video enhancement tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Download UDM10  <https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl> to data/UDM10

Download REDS dataset from [homepage](https://seungjunnah.github.io/Datasets/reds.html) or you can follow
tools/dataset_converters/reds/README.md

```shell
mkdir -p data/
ln -s ${REDS_DATASET_PATH} data/REDS
python tools/dataset_converters/reds/crop_sub_images.py --data-root ./data/REDS # cut REDS images into patches for fas
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
pip install albumentations av==12.0.0
```

## Model Training

```shell
# Training on single card
python3 tools/train.py configs/real_basicvsr/realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py

# Mutiple GPUs on one machine
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/real_basicvsr/realbasicvsr_wogan-c64b20-2x30x8_8xb2-lr1e-4-300k_reds.py 8
```

## References

- [mmagic](https://github.com/open-mmlab/mmagic)

# BiSeNetV2

## Model Description

BiSeNet V2 is a two-pathway architecture for real-time semantic segmentation. One pathway is designed to capture the
spatial details with wide channels and shallow layers, called Detail Branch. In contrast, the other pathway is
introduced to extract the categorical semantics with narrow channels and deep layers, called Semantic Branch. The
Semantic Branch simply requires a large receptive field to capture semantic context, while the detail information can be
supplied by the Detail Branch. Therefore, the Semantic Branch can be made very lightweight with fewer channels and a
fast-downsampling strategy. Both types of feature representation are merged to construct a stronger and more
comprehensive feature representation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

Download cityscapes from [website](https://www.cityscapes-dataset.com/)

```bash
mv /path/to/leftImg8bit_trainvaltest.zip datasets/cityscapes
mv /path/to/gtFine_trainvaltest.zip datasets/cityscapes
cd datasets/cityscapes
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirement
```

## Model Training

```bash
bash train.sh 8 configs/bisenetv2_city.py
```

## Model Results

| Model     | FPS         | ss    | ssc   | msf   | mscf  |
|-----------|-------------|-------|-------|-------|-------|
| BiSeNetV2 | 156.81 s/it | 73.42 | 74.67 | 74.99 | 75.71 |

## References

- [BiSeNet](https://github.com/CoinCheung/BiSeNet/tree/master)

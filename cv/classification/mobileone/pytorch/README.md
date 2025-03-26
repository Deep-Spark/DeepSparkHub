# MobileOne

## Model Description

MobileOne is an efficient neural network backbone designed for mobile devices, focusing on real-world latency rather
than just FLOPs or parameter count. It uses reparameterization with depthwise and pointwise convolutions, optimizing for
speed on mobile chips. Achieving under 1ms inference time on iPhone 12 with 75.9% ImageNet accuracy, MobileOne
outperforms other efficient architectures in both speed and accuracy. It's versatile for tasks like image
classification, object detection, and segmentation, making it ideal for mobile deployment.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Prepare your dataset according to the
[docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

The ImageNet dataset path structure should look like:

```bash
imagenet
├── train
│   └── n01440764
│       ├── n01440764_10026.JPEG
│       └── ...
├── train_list.txt
├── val
│   └── n01440764
│       ├── ILSVRC2012_val_00000293.JPEG
│       └── ...
└── val_list.txt
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# clone mmpretrain
cd deepsparkhub/cv/classification/mobileone/pytorch
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
git checkout 4d1dbafaa28af29f5cb907588c019ae4878c2d24

pip3 install -r requirements.txt

## To avoid errors, let's disable version assert temporarily.
sed -i '9,26s/^/# /' mmpretrain/__init__.py

## using python3
sed -i 's/python /python3 /g' tools/dist_train.sh

# install mmpretrain
python3 setup.py install
```

## Model Training

```bash
bash tools/dist_train.sh configs/mobileone/mobileone-s0_8xb32_in1k.py 8
```

## Model Results

| Model     | GPU        | FPS  | TOP1 Accuracy |
|-----------|------------|------|---------------|
| MobileOne | BI-V100 x8 | 1014 | 71.49         |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain/)

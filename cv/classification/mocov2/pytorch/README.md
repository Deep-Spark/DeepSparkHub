# MoCoV2

## Model Description

MoCoV2 is an improved version of Momentum Contrast (MoCo) for unsupervised learning, combining the strengths of
contrastive learning with momentum-based updates. It introduces an MLP projection head and enhanced data augmentation
techniques to boost performance without requiring large batch sizes. This approach enables effective feature learning
from unlabeled data, establishing strong baselines for self-supervised learning. MoCoV2 outperforms previous methods
like SimCLR while maintaining computational efficiency, making it accessible for various computer vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).

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
cd deepsparkhub/cv/classification/mocov2/pytorch
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
# get mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth
wget https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth

# config parameters
vim configs/mocov2/benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint='./mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth', prefix='backbone.')))

bash tools/dist_train.sh configs/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py 8
```

## Model Results

 | Model  | GPU        | FPS  | TOP1 Accuracy |
 |--------|------------|------|---------------|
 | MoCoV2 | BI-V100 x8 | 4663 | 67.50         |

## References

- [mmpretrain](https://github.com/open-mmlab/mmpretrain/)

# BYOL

## Model Description

BYOL (Bootstrap Your Own Latent) is a self-supervised learning method that learns visual representations without
negative samples. It uses two neural networks - an online network and a target network - that learn from each other
through contrasting augmented views of the same image. BYOL's unique approach eliminates the need for negative pairs,
achieving state-of-the-art performance in unsupervised learning. It's particularly effective for pre-training models on
large datasets before fine-tuning for specific tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Prepare your dataset according to the
[docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset). Sign up and login
in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole
ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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
cd deepsparkhub/cv/classification/byol/pytorch
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
mkdir -p data
ln -s /path/to/imagenet data/imagenet

wget https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth
vim configs/byol/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py
model = dict(
    backbone=dict(
        frozen_stages=4,init_cfg=dict(type='Pretrained', checkpoint='./byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth', prefix='backbone.')))
bash tools/dist_train.sh configs/byol/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py 8
```

## Model Results

| Model | GPU        | FPS  | TOP1 Accuracy |
|-------|------------|------|---------------|
| BYOL  | BI-V100 x8 | 5408 | 71.80         |

## References

- [Paper](https://arxiv.org/abs/2006.07733)
- [mmpretrain](https://github.com/open-mmlab/mmpretrain/)

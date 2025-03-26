# WSLD

## Model Description

WSLD (Weighted Soft Label Distillation) is a knowledge distillation technique that focuses on transferring soft label
information from a teacher model to a student model. Unlike traditional distillation methods that use uniform weighting,
WSLD assigns different weights to each class based on their importance or difficulty, allowing the student to focus more
on challenging or critical classes. This approach improves the student model's performance, particularly in imbalanced
datasets or tasks where certain classes require more attention.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

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
yum install mesa-libGL

# Install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/

# Install requirements
pip3 install opencv-python lmdb msgpack
```

### Preprocess Data

The code is used for training Imagenet. Our pre-trained teacher models are Pytorch official models. By default, we pack
the ImageNet data as the lmdb file for faster IO. The lmdb files can be made as follows.

```bash
# 1. Generate the list of the image data.
python3 dataset/mk_img_list.py --image_path /path/to/imagenet --output_path /path/to/imagenet

# 2. Use the image list obtained above to make the lmdb file.
python3 dataset/img2lmdb.py --image_path /path/to/imagenet --list_path /path/to/imagenet --output_path '/path/to/imagenet' --split 'train'
python3 dataset/img2lmdb.py --image_path /path/to/imagenet --list_path /path/to/imagenet --output_path '/path/to/imagenet' --split 'val'
```

## Model Training

- train_with_distillation.py: train the model with our distillation method.
- imagenet_train_cfg.py: all dataset and hyperparameter settings.
- knowledge_distiller.py: our weighted soft label distillation loss.

**Hint**: Set up 'data.data_path' to be '/path/to/imagenet' in 'imagenet_train_cfg.py' before you start training.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 train_with_distillation.py
```

## Model Results

| GPU        | Network   | Method   | acc   |
|------------|-----------|----------|-------|
| BI-V100 x8 | ResNet 34 | Teacher  | 73.19 |
| BI-V100 x8 | ResNet 18 | Original | 69.75 |
| BI-V100 x8 | ResNet 18 | Proposed | 71.6  |

## References

- [Paper](https://arxiv.org/abs/2102.00650)
- [Weighted Soft Label Distillation](https://github.com/bellymonster/Weighted-Soft-Label-Distillation)

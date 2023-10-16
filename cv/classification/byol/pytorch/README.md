# BYOL

> [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

## Model description

**B**ootstrap **Y**our **O**wn **L**atent (BYOL) is a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network.


## Step 1: Installation

```bash
## install libGL
yum install -y mesa-libGL

## install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/
```

```bash
# install mmcv
pushd ../../../../toolbox/MMDetection/patch/mmcv/v2.0.0rc4/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
popd

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
pip3 install mmengine==0.8.3
python3 setup.py install
```

## Step 2: Preparing datasets

Prepare your dataset according to the [docs](https://mmpretrain.readthedocs.io/en/latest/user_guides/dataset_prepare.html#prepare-dataset).
Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. 
Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

## Step 3: Training

```bash
wget https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth
vim configs/byol/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py
model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint='./byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth', prefix='backbone.')))
bash tools/dist_train.sh configs/byol/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py 8
```

## Results
|     GPUs     | FPS       | TOP1 Accuracy  |
| ------------ | --------- | -------------- |
|  BI-V100 x8  |  5408     | 71.80          |


## Reference
- [mmpretrain](https://github.com/open-mmlab/mmpretrain/)


# MoCoV2

> [Improved Baselines with Momentum Contrastive Learning](https://arxiv.org/abs/2003.04297)


## Model description

Contrastive unsupervised learning has recently shown encouraging progress, e.g., in Momentum Contrast (MoCo) and SimCLR. In this note, we verify the effectiveness of two of SimCLR’s design improvements by implementing them in the MoCo framework. With simple modifications to MoCo—namely, using an MLP projection head and more data augmentation—we establish stronger baselines that outperform SimCLR and do not require large training batches. We hope this will make state-of-the-art unsupervised learning research more accessible.

## Step 1: Installation

```bash
## install libGL
yum install mesa-libGL

## install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/
```

```bash
cd deepsparkhub/cv/distiller/CWD/pytorch/mmcv
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
cd deepsparkhub/cv/classification/mocov2/pytorch
git clone -b main https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
git checkout 4d1dbafaa28af29f5cb907588c019ae4878c2d24
pip3 install -r requirements.txt
"To avoid errors, let's disable these lines temporarily."
https://github.com/open-mmlab/mmpretrain/blob/main/mmpretrain/__init__.py#L9C2-L26C36
vim tools/dist_train.sh
python > python3
pip3 install mmengine==0.8.3
python3 setup.py develop 
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

```shell
wget https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth
vim configs/mocov2/benchmarks/resnet50_8xb32-linear-steplr-100e_in1k.py

model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint='./mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth', prefix='backbone.')))

bash tools/dist_train.sh configs/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py 8
```

## Results
|     Model    | FPS (BI x 8)| TOP1 Accuracy |
| ------------ |  ---------  |--------------:|
|    MoCoV2    |  4663       |    67.50      |
## Reference
https://github.com/open-mmlab/mmpretrain/

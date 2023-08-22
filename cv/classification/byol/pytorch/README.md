# BYOL

> [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

## Model description

**B**ootstrap **Y**our **O**wn **L**atent (BYOL) is a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network.


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
wget https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth
vim configs/byol/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py
model = dict(
    backbone=dict(
        frozen_stages=4,
        init_cfg=dict(type='Pretrained', checkpoint='./byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth', prefix='backbone.')))
bash tools/dist_train.sh configs/byol/benchmarks/resnet50_8xb512-linear-coslr-90e_in1k.py 8
```

## Results
|     Model    | FPS (BI x 8)| TOP1 Accuracy |
| ------------ |  ---------  |--------------:|
|    byol      |  5408       |    71.80      |


## Reference
https://github.com/open-mmlab/mmpretrain/


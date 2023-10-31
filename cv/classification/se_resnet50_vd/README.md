# SE_ResNet50_vd

## Model description

The SENet structure is a weighted average between graph channels that can be embedded into other network structures. SE_ResNet50_vd is a model that adds the senet structure to ResNet50, further learning the dependency relationships between graph channels to obtain better image features.

## Step 1: Installation

```
pip3 install -r requirements.txt
python3 -m pip install urllib3==1.26.6
yum install libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64

git clone https://github.com/PaddlePaddle/PaddleClas.git
```

## Step 2: Preparing datasets

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `./PaddleClas/dataset/` to your ImageNet path in later training process.

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

Then move folds under train and val to ILSVRC2012, run:

```
python3 move_data.py
```

finally data path would like:

```
ILSVRC2012
├── n01440764
│       ├── ILSVRC2012_val_00000293.JPEG
│       └── ...
├── n01443537
│       ├── ILSVRC2012_val_00000236.JPEG
│       └── ...
├── train_list.txt
└── val_list.txt
```

## Step 3: Training

```
cd PaddleClas
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./ppcls/configs/ImageNet/SENet/SE_ResNet50_vd.yaml
```

## Results

| GPUS | ACC | FPS |
| ---- | --- | --- |
|      |     |     |

## Reference

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5)

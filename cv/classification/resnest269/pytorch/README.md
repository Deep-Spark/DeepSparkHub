# ResNeSt269
## Model description
A ResNest is a variant on a ResNet, which instead stacks Split-Attention blocks. The cardinal group representations are then concatenated along the channel dimension.As in standard residual blocks, the final output  of otheur Split-Attention block is produced using a shortcut connection.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

:beers: Done!

## Step 2: Training
### Multiple GPUs on one machine
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_resnest269_amp_dist.sh
```

:beers: Done!


## Reference
https://github.com/zhanghang1989/ResNeSt

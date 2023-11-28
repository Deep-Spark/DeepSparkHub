# SEResNeXt
## Model description
SE ResNeXt is a variant of a ResNext that employs squeeze-and-excitation blocks to enable the network to perform dynamic channel-wise feature recalibration.
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


## Step 2: Training
### Multiple GPUs on one machine
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_seresnext101_32x4d_amp_dist.sh
```



## Reference
https://github.com/osmr/imgclsmob/blob/f2993d3ce73a2f7ddba05da3891defb08547d504/pytorch/pytorchcv/models/seresnext.py#L214

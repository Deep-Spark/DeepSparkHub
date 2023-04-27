# ShuffleNetV2
## Model description
ShuffleNet v2 is a convolutional neural network optimized for a direct metric (speed) rather than indirect metrics like FLOPs. It builds upon ShuffleNet v1, which utilised pointwise group convolutions, bottleneck-like structures, and a channel shuffle operation.
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
bash train_shufflenet_v2_x2_0_amp_dist.sh
```
:beers: Done!


## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification#shufflenet-v2)

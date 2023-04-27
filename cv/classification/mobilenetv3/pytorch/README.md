# MobileNetV3

## Model description
MobileNetV3 is a convolutional neural network that is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm, and then subsequently improved through novel architecture advances. Advances include (1) complementary search techniques, (2) new efficient versions of nonlinearities practical for the mobile setting, (3) new efficient network design.

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
### Multiple GPUs on one machine (AMP)
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_mobilenet_v3_large_dist.sh
```

:beers: Done!

## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small)

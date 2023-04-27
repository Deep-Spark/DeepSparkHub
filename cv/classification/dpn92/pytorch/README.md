# DPN92
## Model description
A Dual Path Network (DPN) is a convolutional neural network which presents a new topology of connection paths internally. The intuition is that ResNets enables feature re-usage while DenseNet enables new feature exploration, and both are important for learning good representations. To enjoy the benefits from both path topologies, Dual Path Networks share common features while maintaining the flexibility to explore new features through dual path architectures.

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

## Step2: Training

### Multiple GPUs on one machine (AMP)
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_dpn92_amp_dist.sh
```

:beers: Done!


## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification)

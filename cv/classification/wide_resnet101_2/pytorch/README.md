# Wide_ResNet101_2

## Model description
Wide Residual Networks are a variant on ResNets where we decrease depth and increase the width of residual networks. This is achieved through the use of wide residual blocks.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```
:beers: Done!

## Step 2: Training
### Multiple GPUs on one machine
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_wide_resnet101_2_amp_dist.sh
```

:beers: Done!


## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification)

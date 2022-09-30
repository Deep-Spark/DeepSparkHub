# ResNeSt101

## Model description
A ResNest is a variant on a ResNet, which instead stacks Split-Attention blocks. The cardinal group representations are then concatenated along the channel dimension.As in standard residual blocks, the final output  of otheur Split-Attention block is produced using a shortcut connection.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```
:beers: Done!

## Step 2: Training
### Multiple GPUs on one machine (AMP)
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_resnest101_amp_dist.sh
```

:beers: Done!


## Reference
https://github.com/zhanghang1989/ResNeSt

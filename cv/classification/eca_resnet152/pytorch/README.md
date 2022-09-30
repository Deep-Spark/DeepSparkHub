# ECA_RESNET152
## Model description
An ECA-Net is a type of convolutional neural network that utilises an Efficient Channel Attention module.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```
:beers: Done!

## Step 2: Training
### Multiple GPUs on one machine (AMP)
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_eca_resnet152_amp_dist.sh
```

:beers: Done!


## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification)

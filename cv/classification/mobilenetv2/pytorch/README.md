# MobileNetV2

## Model description
MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers. The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```
:beers: Done!

## Step 2: Training
### Multiple GPUs on one machine (AMP)
Set data path by `export DATA_PATH=/path/to/imagenet`. The following command uses all cards to train:

```bash
bash train_mobilenet_v2_amp_dist.sh
```

:beers: Done!


## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification)

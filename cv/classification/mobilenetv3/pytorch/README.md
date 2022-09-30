# MobileNetV3

## Model description
MobileNetV3 is a convolutional neural network that is tuned to mobile phone CPUs through a combination of hardware-aware network architecture search (NAS) complemented by the NetAdapt algorithm, and then subsequently improved through novel architecture advances. Advances include (1) complementary search techniques, (2) new efficient versions of nonlinearities practical for the mobile setting, (3) new efficient network design.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
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

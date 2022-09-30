# ShuffleNetV2
## Model description
ShuffleNet v2 is a convolutional neural network optimized for a direct metric (speed) rather than indirect metrics like FLOPs. It builds upon ShuffleNet v1, which utilised pointwise group convolutions, bottleneck-like structures, and a channel shuffle operation.
## Step 1: Installing
```bash
pip3 install -r requirements.txt
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

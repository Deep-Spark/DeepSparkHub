# ResNet50
## Model description
Residual Networks, or ResNets, learn residual functions with reference to the layer inputs, instead of learning unreferenced functions. Instead of hoping each few stacked layers directly fit a desired underlying mapping, residual nets let these layers fit a residual mapping.

## Step 1: Installing
```bash
pip3 install -r requirements.txt
```
:beers: Done!

## Step 2: Training

### One single GPU
```bash
bash fp32_1card.sh --data-path /path/to/imagenet
```
### One single GPU (AMP)
```bash
bash amp_1card.sh --data-path /path/to/imagenet
```
### Multiple GPUs on one machine
```bash
bash fp32_4card.sh --data-path /path/to/imagenet
bash fp32_8card.sh --data-path /path/to/imagenet
```
### Multiple GPUs on one machine (AMP)
```bash
bash amp_4card.sh --data-path /path/to/imagenet
bash amp_8card.sh --data-path /path/to/imagenet
```
### Multiple GPUs on two machines
```bash
bash fp32_16card.sh --data-path /path/to/imagenet
```

## Results on BI-V100

|             | FP32                                            | AMP+NHWC                                      |
| ----------- | ----------------------------------------------- | --------------------------------------------- |
| single card | Acc@1=76.02,FPS=330,Time=4d3h，BatchSize=280    | Acc@1=75.56,FPS=550,Time=2d13h，BatchSize=300 |
| 4 cards     | Acc@1=75.89,FPS=1233,Time=1d2h，BatchSize=300   | Acc@1=79.04,FPS=2400,Time=11h，BatchSize=512  |
| 8 cards     | Acc@1=74.98,FPS=2150,Time=12h43m，BatchSize=300 | Acc@1=76.43,FPS=4200,Time=8h，BatchSize=480   |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| top1 75.9%           | SDK V2.2,bs:512,8x,AMP                   | 5221        | 76.43%   | 128\*8     | 0.97        | 29.1\*8                 | 1         |


## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification)

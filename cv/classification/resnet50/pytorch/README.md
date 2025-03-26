# ResNet50

## Model Description

ResNet50 is a deep convolutional neural network with 50 layers, known for its innovative residual learning framework. It
introduces skip connections that bypass layers, enabling the training of very deep networks by addressing vanishing
gradient problems. This architecture achieved breakthrough performance in image classification tasks, winning the 2015
ImageNet competition. ResNet50's efficient design and strong feature extraction capabilities make it widely used in
computer vision applications, serving as a backbone for various tasks like object detection and segmentation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

## Model Training

```bash
# One single GPU
bash scripts/fp32_1card.sh --data-path /path/to/imagenet

# One single GPU (AMP)
bash scripts/amp_1card.sh --data-path /path/to/imagenet

# Multiple GPUs on one machine
bash scripts/fp32_4cards.sh --data-path /path/to/imagenet
bash scripts/fp32_8cards.sh --data-path /path/to/imagenet

# Multiple GPUs on one machine (AMP)
bash scripts/amp_4cards.sh --data-path /path/to/imagenet
bash scripts/amp_8cards.sh --data-path /path/to/imagenet

### Multiple GPUs on two machines
bash scripts/fp32_16cards.sh --data-path /path/to/imagenet
```

## Model Results

| Model    | GPU        | FP32                                            | AMP+NHWC                                      |
|----------|------------|-------------------------------------------------|-----------------------------------------------|
| ResNet50 | BI-V100 x1 | Acc@1=76.02,FPS=330,Time=4d3h，BatchSize=280    | Acc@1=75.56,FPS=550,Time=2d13h，BatchSize=300 |
| ResNet50 | BI-V100 x4 | Acc@1=75.89,FPS=1233,Time=1d2h，BatchSize=300   | Acc@1=79.04,FPS=2400,Time=11h，BatchSize=512  |
| ResNet50 | BI-V100 x8 | Acc@1=74.98,FPS=2150,Time=12h43m，BatchSize=300 | Acc@1=76.43,FPS=4200,Time=8h，BatchSize=480   |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| top1 75.9%           | SDK V2.2,bs:512,8x,AMP                   | 5221        | 76.43%   | 128\*8     | 0.97        | 29.1\*8                 | 1         |

## References

- [vision](https://github.com/pytorch/vision/tree/main/references/classification)

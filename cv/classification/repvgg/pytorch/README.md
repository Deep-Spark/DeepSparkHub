# RepVGG

## Model Description

RepVGG is a simple yet powerful convolutional neural network architecture that combines training-time multi-branch
topology with inference-time VGG-like simplicity. It uses structural re-parameterization to convert complex training
models into efficient inference models composed solely of 3x3 convolutions and ReLU activations. This approach achieves
state-of-the-art performance in image classification tasks while maintaining high speed and efficiency. RepVGG's design
is particularly suitable for applications requiring both high accuracy and fast inference, making it ideal for
real-world deployment scenarios.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.03  |

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

### Install Dependencies

```bash
pip3 install timm yacs
git clone https://github.com/DingXiaoH/RepVGG.git
cd RepVGG
git checkout eae7c5204001eaf195bbe2ee72fb6a37855cce33
```

## Model Training

```bash
# fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' main.py

# change dataset load
# Tips: "import os" into data/build.py
sed -i "s@dataset = torchvision.datasets.ImageNet(root=config.DATA.DATA_PATH, split='train' if is_train else 'val', transform=transform)@dataset = datasets.ImageFolder(os.path.join(config.DATA.DATA_PATH, prefix), transform=transform)@" data/build.py

python3 -m torch.distributed.launch --nproc_per_node 4 --master_port 12349 main.py --arch RepVGG-A0 --data-path ./imagenet --batch-size 32 --tag train_from_scratch --output ./ --opts TRAIN.EPOCHS 300 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 1e-4 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET weak AUG.MIXUP 0.0 DATA.DATASET imagenet DATA.IMG_SIZE 224
```

The original RepVGG models were trained in 120 epochs with cosine learning rate decay from 0.1 to 0. We used 8 GPUs,
global batch size of 256, weight decay of 1e-4 (no weight decay on fc.bias, bn.bias, rbr_dense.bn.weight and
rbr_1x1.bn.weight) (weight decay on rbr_identity.weight makes little difference, and it is better to use it in most of
the cases), and the same simple data preprocssing as the PyTorch official example:

```py
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
```

The valid model names include (--arch [model name]):

RepVGGplus-L2pse, RepVGG-A0, RepVGG-A1, RepVGG-A2, RepVGG-B0, RepVGG-B1, RepVGG-B1g2, RepVGG-B1g4, RepVGG-B2,
RepVGG-B2g2, RepVGG-B2g4, RepVGG-B3, RepVGG-B3g2, RepVGG-B3g4.

| Model     | GPU        | FP32         |
|-----------|------------|--------------|
| RepVGG-A0 | BI-V100 x8 | Acc@1=0.7241 |

## References

- [RepVGG](https://github.com/DingXiaoH/RepVGG/tree/eae7c5204001eaf195bbe2ee72fb6a37855cce33)

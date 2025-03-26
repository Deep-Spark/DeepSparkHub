# ConvNext

## Model Description

ConvNext is a modern convolutional neural network architecture that bridges the gap between traditional ConvNets and
Vision Transformers. Inspired by Transformer designs, it incorporates techniques like large kernel sizes, layer
normalization, and inverted bottlenecks to achieve state-of-the-art performance. ConvNext demonstrates that properly
modernized ConvNets can match or exceed Transformer-based models in accuracy and efficiency across various vision tasks.
Its simplicity and strong performance make it a compelling choice for image classification and other computer vision
applications.

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

### Install Dependencies

```bash
pip install timm==0.4.12 tensorboardX six torch torchvision

git clone https://github.com/facebookresearch/ConvNeXt.git
cd ConvNeXt/
git checkout 048efcea897d999aed302f2639b6270aedf8d4c8
```

## Model Training

```bash
# Multiple GPUs on one machine
python3 -m torch.distributed.launch --nproc_per_node=8 main.py \
                                    --model convnext_tiny \
                                    --drop_path 0.1 \
                                    --batch_size 128 \
                                    --lr 4e-3 \
                                    --update_freq 4 \
                                    --model_ema true \
                                    --model_ema_eval true \
                                    --data_path /path/to/imagenet \
                                    --output_dir /path/to/save_results
```

## References

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)

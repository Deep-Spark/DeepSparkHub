# Swin Transformer

## Model Description

The Swin Transformer is a hierarchical vision transformer that introduces shifted windows for efficient self-attention
computation. It processes images in local windows, reducing computational complexity while maintaining global modeling
capabilities. The architecture builds hierarchical feature maps by merging image patches in deeper layers, making it
suitable for both image classification and dense prediction tasks. Swin Transformer achieves state-of-the-art
performance in various vision tasks, offering a powerful alternative to traditional convolutional networks with its
transformer-based approach.

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
git clone https://github.com/microsoft/Swin-Transformer.git
git checkout f82860bfb5225915aca09c3227159ee9e1df874d
cd Swin-Transformer
pip install timm==0.4.12 yacs
```

## Model Training

```bash
# Multiple GPUs on one machine

## fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' main.py

python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path /path/to/imagenet --batch-size 128
```

## References

- [Swin-Transformer](https://github.com/microsoft/Swin-Transformer)

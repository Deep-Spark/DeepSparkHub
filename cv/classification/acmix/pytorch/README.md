# ACmix

## Model Description

ACmix is an innovative deep learning model that unifies convolution and self-attention mechanisms by revealing their
shared computational foundation. It demonstrates that both operations can be decomposed into 1x1 convolutions followed
by different aggregation strategies. This insight enables ACmix to efficiently combine the benefits of both paradigms -
the local feature extraction of convolutions and the global context modeling of self-attention. The model achieves
improved performance on image recognition tasks with minimal computational overhead compared to pure convolution or
attention-based approaches.

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
git clone https://github.com/LeapLabTHU/ACmix.git
pip install termcolor==1.1.0 yacs==0.1.8 timm==0.4.5
cd ACmix/Swin-Transformer
git checkout 81dddb6dff98f5e238a7fb6ab174e256489c07fa
```

## Model Training

```bash
# Swin-S + ACmix on ImageNet using 8 cards

## fix --local-rank for torch 2.x
sed -i 's/--local_rank/--local-rank/g' main.py

python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 main.py --cfg configs/acmix_swin_small_patch4_window7_224.yaml --data-path /path/to/imagenet --batch-size 128
```

## Model Results

| Model | GPU     | batch_size | Single Card | 8 Cards |
|-------|---------|------------|-------------|---------|
| ACmix | BI-V100 | 128        | 63.59       | 502.22  |

## References

- [acmix](https://github.com/leaplabthu/acmix)

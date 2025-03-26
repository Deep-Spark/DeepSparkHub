# L-Verse

## Model Description

L-Verse is an innovative vision-language model that bridges image and text through its unique architecture. It combines
a Feature-Augmented Variational Autoencoder (AugVAE) with a Bidirectional Auto-Regressive Transformer (BiART) to enable
seamless image-to-text and text-to-image generation. Unlike traditional models, L-Verse excels in both directions
without requiring fine-tuning or additional frameworks. Its AugVAE component achieves state-of-the-art image
reconstruction, while BiART effectively distinguishes between conditional references and generation targets. L-Verse
demonstrates impressive results in multimodal tasks, particularly on MS-COCO Captions dataset.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
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
pip3 install pudb "pytorch-lightning==1.5" einops regex ftfy cython webdataset==0.2.20 pillow wandb scikit-learn tensorboard
```

## Model Training

AugVAE(AugVAE-ML)

```bash
git clone https://github.com/tgisaturday/L-Verse.git
cd /path/to/L-Verse/pytorch
git checkout 504a6bf740812bdd2022f31f969968ec31794033
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python3 train_vae.py --config ./configs/imagenet_augvae_ml.yaml --train_dir /path/to/imagenet/train --val_dir /path/to/imagenet/val --gpus 8 --batch_size 4 --epochs 2
```

## References

- [L-Verse](https://github.com/tgisaturday/L-Verse)
# MAE

## Model Description

MAE (Masked Autoencoders) is a self-supervised learning model for computer vision that learns powerful representations
by reconstructing randomly masked portions of input images. It employs an asymmetric encoder-decoder architecture where
the encoder processes only the visible patches, and the lightweight decoder reconstructs the original image from the
latent representation and mask tokens. MAE demonstrates that high masking ratios (e.g., 75%) can lead to robust feature
learning, making it scalable and effective for various downstream vision tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

Download dataset.

```bash
cd /home/datasets/cv/ImageNet_ILSVRC2012
Download the [ImageNet Dataset](https://www.image-net.org/download.php)
```

Download pretrain weight

```bash
cd pretrain
Download the [pretrain_mae_vit_base_mask_0.75_400e.pth](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa)
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
mkdir -p /home/datasets/cv/ImageNet_ILSVRC2012
mkdir -p pretrain
mkdir -p output
```

## Model Training

```bash
# Finetune
cd ..
bash run.sh
```

## Model Results

| GPU        | FPS  | Train Epochs | Accuracy |
|------------|------|--------------|----------|
| BI-V100 x8 | 1233 | 100          | 82.9%    |

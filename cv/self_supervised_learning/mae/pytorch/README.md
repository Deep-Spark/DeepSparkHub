# MAE

## Model Description

MAE (Masked Autoencoders) is a self-supervised learning model for computer vision that learns powerful representations
by reconstructing randomly masked portions of input images. It employs an asymmetric encoder-decoder architecture where
the encoder processes only the visible patches, and the lightweight decoder reconstructs the original image from the
latent representation and mask tokens. MAE demonstrates that high masking ratios (e.g., 75%) can lead to robust feature
learning, making it scalable and effective for various downstream vision tasks.

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

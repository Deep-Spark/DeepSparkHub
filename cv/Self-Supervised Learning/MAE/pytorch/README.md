# MAE-pytorch

## Model description
This repository is built upon BEiT, an unofficial PyTorch implementation of Masked Autoencoders Are Scalable Vision Learners. We implement the pretrain and finetune process according to the paper, but still can't guarantee the performance reported in the paper can be reproduced!

## Environment

```
cd MAE-pytorch
pip3 install -r requirements.txt
mkdir -p /home/datasets/cv/ImageNet_ILSVRC2012
mkdir -p pretrain
mkdir -p output
```

## Download dataset

```
cd /home/datasets/cv/ImageNet_ILSVRC2012
Download the [ImageNet Dataset](https://www.image-net.org/download.php)
```

## Download pretrain weight

```
cd pretrain
Download the [pretrain_mae_vit_base_mask_0.75_400e.pth](https://drive.google.com/drive/folders/182F5SLwJnGVngkzguTelja4PztYLTXfa)
```

## Finetune

```
bash run.sh
```

## Results on BI-V100

```
| GPUs | FPS | Train Epochs | Accuracy  |
|------|-----|--------------|------|
| 1x8  | 1233 | 100           | 82.9% |
```


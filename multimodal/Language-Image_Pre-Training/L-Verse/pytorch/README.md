# L-Verse

## Model description

Far beyond learning long-range interactions of natural language, transformers are becoming the de-facto standard for many vision tasks with their power and scalability. Especially with cross-modal tasks between image and text, vector quantized variational autoencoders (VQ-VAEs) are widely used to make a raw RGB image into a sequence of feature vectors. To better leverage the correlation between image and text, we propose L-Verse, a novel architecture consisting of feature-augmented variational autoencoder (AugVAE) and bidirectional auto-regressive transformer (BiART) for image-to-text and text-to-image generation. Our AugVAE shows the state-of-the-art reconstruction performance on ImageNet1K validation set, along with the robustness to unseen images in the wild. Unlike other models, BiART can distinguish between image (or text) as a conditional reference and a generation target. L-Verse can be directly used for image-to-text or text-to-image generation without any finetuning or extra object detection framework. In quantitative and qualitative experiments, L-Verse shows impressive results against previous methods in both image-to-text and text-to-image generation on MS-COCO Captions. We furthermore assess the scalability of L-Verse architecture on Conceptual Captions and present the initial result of bidirectional vision-language representation learning on general domain.

## Step 1: Installing packages

```
pip3 install pudb "pytorch-lightning==1.5" einops regex ftfy cython webdataset==0.2.20 pillow wandb sklearn tensorboard
```

## Step 2: Preparing datasets
* Download ImageNet dataset and place it in `/home/datasets/cv/imagenet-mini` as follows:

```
├── imagenet-mini
│   ├── train
|   |   |── n01440764
|   |   |── n01734418
|   |   |── ......
│   ├── val
|   |   |── n01440764
|   |   |── n01734418
|   |   |── ......
```

## Step 3: Training AugVAE(AugVAE-ML)

```
$ cd /path/to/L-Verse/pytorch
$ CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_vae.py --config ./configs/imagenet_augvae_ml.yaml --train_dir /home/datasets/cv/imagenet-mini/train --val_dir /home/datasets/cv/imagenet-mini/val --gpus 4 --batch_size 4 --epochs 2
```

## Reference
https://github.com/tgisaturday/L-Verse
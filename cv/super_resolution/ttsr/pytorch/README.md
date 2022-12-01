# TTSR

## Model description

We study on image super-resolution (SR), which aims to recover realistic textures from a low-resolution (LR) image. Recent progress has been made by taking high-resolution images as references (Ref), so that relevant textures can be transferred to LR images. However, existing SR approaches neglect to use attention mechanisms to transfer high-resolution (HR) textures from Ref images, which limits these approaches in challenging cases. In this paper, we propose a novel Texture Transformer Network for Image Super-Resolution (TTSR), in which the LR and Ref images are formulated as queries and keys in a transformer, respectively. TTSR consists of four closely-related modules optimized for image generation tasks, including a learnable texture extractor by DNN, a relevance embedding module, a hard-attention module for texture transfer, and a soft-attention module for texture synthesis. Such a design encourages joint feature learning across LR and Ref images, in which deep feature correspondences can be discovered by attention, and thus accurate texture features can be transferred. The proposed texture transformer can be further stacked in a cross-scale way, which enables texture recovery from different levels (e.g., from 1x to 4x magnification). Extensive experiments show that TTSR achieves significant improvements over state-of-the-art approaches on both quantitative and qualitative evaluations.


## Step 1: Installing packages

```bash
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```bash
mkdir -p data/
cd data
# Download CUFED Dataset from [homepage](https://zzutk.github.io/SRNTT-Project-Page)
# the folder would be like:
data/CUFED/
└── train
    ├── input
    └── ref
```

## Step 3: Training

### Multiple GPUs on one machine

```bash
CUDA_VISIBLE_DEVICES=${gpu_id_1,gpu_id_2,...} bash train.sh ${num_gpus}
```

For example, GPU 5 and GPU 7 are available for use and you can use these two GPUs as follows:

```bash
CUDA_VISIBLE_DEVICES=5,7 bash train.sh 2
```

## Reference
https://github.com/open-mmlab/mmediting

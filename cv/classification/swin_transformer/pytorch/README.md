# Swin Transformer
## Model description
The Swin Transformer is a type of Vision Transformer. It builds hierarchical feature maps by merging image patches (shown in gray) in deeper layers and has linear computation complexity to input image size due to computation of self-attention only within each local window (shown in red). It can thus serve as a general-purpose backbone for both image classification and dense recognition tasks.

## Step 1: Installing

```bash
pip install timm==0.4.12
pip install yacs
```

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

## Step 2: Training
### Multiple GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
    --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path /path/to/imagenet --batch-size 128
```

## Reference
https://github.com/microsoft/Swin-Transformer

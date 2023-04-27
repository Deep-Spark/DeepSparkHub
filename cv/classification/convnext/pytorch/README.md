# ConvNext

## Model description
The ConvNeXT model was proposed in [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) by Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie. ConvNeXT is a pure convolutional model (ConvNet), inspired by the design of Vision Transformers, that claims to outperform them.

## Step 1: Installing
```bash
pip install timm==0.4.12 tensorboardX six torch torchvision
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

## Reference
https://github.com/facebookresearch/ConvNeXt

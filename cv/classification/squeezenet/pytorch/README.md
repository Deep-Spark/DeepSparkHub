# SqueezeNet

## Model description
SqueezeNet is a convolutional neural network that employs design strategies to reduce the number of parameters, notably with the use of fire modules that "squeeze" parameters using 1x1 convolutions.

## Step 1: Installing
```bash
pip3 install torch torchvision
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

:beers: Done!

## Step 2: Training
### One single GPU
```bash
python3 train.py --data-path /path/to/imagenet --model squeezenet1_0 --lr 0.001
```
### Multiple GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenette --model squeezenet1_0 --lr 0.001
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py

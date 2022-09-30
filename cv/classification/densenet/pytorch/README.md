# DenseNet
## Model description
A DenseNet is a type of convolutional neural network that utilises dense connections between layers, through Dense Blocks, where we connect all layers (with matching feature-map sizes) directly with each other. To preserve the feed-forward nature, each layer obtains additional inputs from all preceding layers and passes on its own feature-maps to all subsequent layers.

## Step 1: Installing
```bash
pip install torch torchvision
```
## Step 2: Training
### One single GPU
```bash
python3 train.py --data-path /home/datasets/cv/imagenet --model densenet201 --batch-size 128
```
### Multiple GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /home/datasets/cv/imagenet --model densenet201 --batch-size 128
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py

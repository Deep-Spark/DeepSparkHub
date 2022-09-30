# Xception

## Model description
Xception is a convolutional neural network architecture that relies solely on depthwise separable convolution layers.

## Step 1: Installing
```bash
pip3 install torch torchvision
```

## Step 2: Training
### One single GPU
```bash
python3 train.py --data-path /home/datasets/cv/imagenet --model xception
```
### Multiple GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /home/datasets/cv/imagenet --model xception
```

## Reference
https://github.com/tstandley/Xception-PyTorch

# efficientb4
## Model description
EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient.
## Step 1: Installing
```bash
pip3 install torch torchvision
```
:beers: Done!

## Step 2: Training
### One single GPU
```bash
python3 train.py --data-path /home/datasets/cv/imagenet --model efficientnet_b4 --batch-size 128
```
### Multiple GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /home/datasets/cv/imagenet --model efficientnet_b4 --batch-size 128
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

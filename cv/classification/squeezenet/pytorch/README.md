# SqueezeNet

## Model description
SqueezeNet is a convolutional neural network that employs design strategies to reduce the number of parameters, notably with the use of fire modules that "squeeze" parameters using 1x1 convolutions.

## Step 1: Installing
```bash
pip3 install torch torchvision
```
:beers: Done!

## Step 2: Training
### One single GPU
```bash
python3 train.py --data-path /home/datasets/cv/imagenet --model squeezenet1_0 --lr 0.001
```
### Multiple GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /home/datasets/cv/imagenette --model squeezenet1_0 --lr 0.001
```

## Reference
https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py

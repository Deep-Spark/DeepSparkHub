# LeNet

## Model description
LeNet is a classic convolutional neural network employing the use of convolutions, pooling and fully connected layers. It was used for the handwritten digit recognition task with the MNIST dataset. The architectural design served as inspiration for future networks such as AlexNet and VGG.

## Step 1: Installing

```bash
pip3 install torch
pip3 install torchvision
```

## Step 2: Training
### One single GPU
```bash
python3 train.py --data-path /home/datasets/cv/imagenet --model lenet 
```
### 8 GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /home/datasets/cv/imagenet --model lenet 
```

## Reference
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

# LeNet

## Model description
LeNet is a classic convolutional neural network employing the use of convolutions, pooling and fully connected layers. It was used for the handwritten digit recognition task with the MNIST dataset. The architectural design served as inspiration for future networks such as AlexNet and VGG.

## Step 1: Installing

```bash
pip3 install torch
pip3 install torchvision
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
### One single GPU
```bash
python3 train.py --data-path /path/to/imagenet --model lenet 
```
### 8 GPUs on one machine
```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenet --model lenet 
```

## Reference
http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf

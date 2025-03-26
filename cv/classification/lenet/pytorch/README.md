# LeNet

## Model Description

LeNet is a pioneering convolutional neural network architecture developed for handwritten digit recognition. It
introduced fundamental concepts like convolutional layers, pooling, and fully connected layers, laying the groundwork
for modern deep learning. Designed for the MNIST dataset, LeNet demonstrated the effectiveness of CNNs for image
recognition tasks. Its simple yet effective architecture inspired subsequent networks like AlexNet and VGG, making it a
cornerstone in the evolution of deep learning for computer vision applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

## Model Training

```bash
# One single GPU
python3 train.py --data-path /path/to/imagenet --model lenet 

# 8 GPUs on one machine
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --data-path /path/to/imagenet --model lenet 
```

## References

- [Paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)

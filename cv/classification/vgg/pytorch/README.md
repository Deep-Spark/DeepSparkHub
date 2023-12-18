# VGG16
## Model description
VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer.


## Step 1: Installation

```bash
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

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

## Step 3: Training

```bash
# Set data path
export DATA_PATH=/path/to/imagenet

# Multiple GPUs on one machine
bash train_vgg16_amp_dist.sh
```
Install zlib-1.2.9 if reports "iZLIB_1.2.9 not found" when run train_vgg16_amp_dist.sh

```bash
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ../
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/
```

## Reference
- [torchvision](https://github.com/pytorch/vision/tree/main/references/classification)

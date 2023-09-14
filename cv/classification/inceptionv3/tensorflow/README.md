# Inception V3

## Model description

Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements including using Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifer to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).

## Step 1: Installation

```bash
pip3 install absl-py git+https://github.com/NVIDIA/dllogger#egg=dllogger
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

Refer below links to convert ImageNet data to TFrecord data.

- [Downloading and converting to TFRecord format](https://github.com/kmonachopoulos/ImageNet-to-TFrecord)
- [TensoFlow](https://github.com/tensorflow/models/tree/master/research/slim#downloading-and-converting-to-tfrecord-format)

Put the TFrecord data in "./imagenet_tfrecord" directory.

## Step 3: Training

```bash
# 1 GPU
bash run_train_inceptionV3_imagenet.sh

# 8 GPUs
bash run_train_inceptionV3_multigpu_imagenet.sh --epoch 200
```

## Results

| GPUS       | acc   | fps          |
| ---------- | ----- | ------------ |
| BI V100×8 | 76.4% | 312 images/s |

## Reference

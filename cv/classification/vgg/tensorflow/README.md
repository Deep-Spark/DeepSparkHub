# VGG16

## Model description

VGG is a classical convolutional neural network architecture. It was based on an analysis of how to increase the depth of such networks. The network utilises small 3 x 3 filters. Otherwise the network is characterized by its simplicity: the only other components being pooling layers and a fully connected layer.

## Step 1: Installation

```bash
pip3 install absl-py git+https://github.com/NVIDIA/dllogger#egg=dllogger
```

## Step 2: Preparing datasets

You can get ImageNet 1K TFrecords ILSVRC2012 dataset directly from below links:
- [ImageNet 1K TFrecords ILSVRC2012 - part 0](https://www.kaggle.com/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0)
- [ImageNet 1K TFrecords ILSVRC2012 - part 1](https://www.kaggle.com/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1)

The ImageNet TFrecords dataset path structure should look like:

```bash
imagenet_tfrecord
├── train-00000-of-01024
├── ...
├── train-01023-of-01024
├── validation-00000-of-00128
├── ...
└── validation-00127-of-00128
```

## Step 3: Training
**Put the TFrecords data in "./imagenet_tfrecord" directory or create a soft link.**
```bash
# 1 GPU
bash run_train_vgg16_imagenet.sh

# 8 GPUs
bash run_train_vgg16_multigpu_imagenet.sh
```

## Results

| GPUS      |    acc                    |   fps |
| ----------| --------------------------| ----- | 
| BI V100×8 | acc@1=0.7160,acc@5=0.9040 | 435.9 |

## Reference
- [TensorFlow/benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)
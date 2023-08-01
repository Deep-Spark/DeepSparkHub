# AlexNet

AlexNet is a groundbreaking convolutional neural network (CNN) introduced in 2012. It revolutionized computer vision by demonstrating the power of deep learning in image classification. With eight layers, including five convolutional and three fully connected layers, it achieved remarkable results on the ImageNet challenge with a top-1 accuracy of around 57.1%. AlexNet's success paved the way for widespread adoption of deep neural networks in computer vision tasks.

## Installation

```bash
pip3 install absl-py git+https://github.com/NVIDIA/dllogger#egg=dllogger
```

## Preparing datasets

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

## Training

**Put the TFrecords data in "./imagenet_tfrecord" directory or create a soft link.**

```bash
# 1 GPU
bash run_train_alexnet_imagenet.sh

# 8 GPUs
bash run_train_alexnet_multigpu_imagenet.sh
```

## Results
|GPUs|ACC|FPS|
|:---:|:---:|:---:|
|BI-v100 x8|Accuracy @1 = 0.5633 Accuracy @ 5 = 0.7964|1833.9 images/sec|

## Reference
- [TensorFlow Models](https://github.com/tensorflow/models) 

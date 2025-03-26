# VGG16

## Model Description

VGG is a classic convolutional neural network architecture known for its simplicity and depth. It uses small 3x3
convolutional filters stacked in multiple layers, allowing for effective feature extraction. The architecture typically
includes 16 or 19 weight layers, with VGG16 being the most popular variant. VGG achieved state-of-the-art performance in
image classification tasks and became a benchmark for subsequent CNN architectures. Its uniform structure and deep
design have influenced many modern deep learning models in computer vision.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

You can get ImageNet 1K TFrecords ILSVRC2012 dataset directly from below links:

- [ImageNet 1K TFrecords ILSVRC2012 - part
  0](https://www.kaggle.com/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-0)
- [ImageNet 1K TFrecords ILSVRC2012 - part
  1](https://www.kaggle.com/datasets/hmendonca/imagenet-1k-tfrecords-ilsvrc2012-part-1)

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

### Install Dependencies

```bash
pip3 install absl-py git+https://github.com/NVIDIA/dllogger#egg=dllogger
```

## Model Training

Put the TFrecords data in "./imagenet_tfrecord" directory or create a soft link.

```bash
# 1 GPU
bash run_train_vgg16_imagenet.sh

# 8 GPUs
bash run_train_vgg16_multigpu_imagenet.sh
```

## Model Results

| Model | GPU        | acc                       | fps   |
|-------|------------|---------------------------|-------|
| VGG16 | BI-V100 ×8 | acc@1=0.7160,acc@5=0.9040 | 435.9 |

## References

- [tensorflow/benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)

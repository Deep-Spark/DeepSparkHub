# AlexNet

## Model Description

AlexNet is a groundbreaking deep convolutional neural network that revolutionized computer vision. It introduced key
innovations like ReLU activations, dropout regularization, and GPU acceleration. With its 8-layer architecture featuring
5 convolutional and 3 fully-connected layers, AlexNet achieved record-breaking performance on ImageNet in 2012. Its
success popularized deep learning and established CNNs as the dominant approach for image recognition. AlexNet's design
principles continue to influence modern neural network architectures in computer vision applications.

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
bash run_train_alexnet_imagenet.sh

# 8 GPUs
bash run_train_alexnet_multigpu_imagenet.sh
```

## Model Results

| Model   | GPU        | ACC                                        | FPS               |
|---------|------------|--------------------------------------------|-------------------|
| AlexNet | BI-v100 x8 | Accuracy @1 = 0.5633 Accuracy @ 5 = 0.7964 | 1833.9 images/sec |

## References

- [TensorFlow/benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)

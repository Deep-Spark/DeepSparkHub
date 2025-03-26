# ResNet50

## Model Description

ResNet50 is a deep convolutional neural network with 50 layers, known for its innovative residual learning framework. It
introduces skip connections that bypass layers, enabling the training of very deep networks by addressing vanishing
gradient problems. This architecture achieved breakthrough performance in image classification tasks, winning the 2015
ImageNet competition. ResNet50's efficient design and strong feature extraction capabilities make it widely used in
computer vision applications, serving as a backbone for various tasks like object detection and segmentation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Download and convert to TFRecord format following [ImageNet-to-TFrecord](https://github.com/kmonachopoulos/ImageNet-to-TFrecord).

Or [here](https://github.com/tensorflow/models/tree/master/research/slim#downloading-and-converting-to-tfrecord-format)

Make a file named imagenet_tfrecord, and store imagenet datasest convert to imagenet_tfrecord

### Install Dependencies

```shell
pip3 install absl-py git+https://github.com/NVIDIA/dllogger#egg=dllogger
```

## Model Training

```shell
ln -s /path/to/imagenette_tfrecord ./imagenette

# Training on single card
bash run_train_resnet50_imagenette.sh

# Training on mutil-cards
bash run_train_resnet50_multigpu_imagenette.sh
```

## Model Results

| Model    | GPU        | acc    | fps   |
|----------|------------|--------|-------|
| ResNet50 | BI-V100 x8 | 0.9860 | 236.9 |

## Reference
- [TensorFlow/benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks)
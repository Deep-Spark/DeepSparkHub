# ResNet50

## Model Description

ResNet50 is a deep convolutional neural network with 50 layers, known for its innovative residual learning framework. It
introduces skip connections that bypass layers, enabling the training of very deep networks by addressing vanishing
gradient problems. This architecture achieved breakthrough performance in image classification tasks, winning the 2015
ImageNet competition. ResNet50's efficient design and strong feature extraction capabilities make it widely used in
computer vision applications, serving as a backbone for various tasks like object detection and segmentation.

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
# Training on single card
bash run_train_resnet50_imagenette.sh

# Training on mutil-cards
bash run_train_resnet50_multigpu_imagenette.sh
```

## Model Results

| Model    | GPU        | acc    | fps   |
|----------|------------|--------|-------|
| ResNet50 | BI-V100 x8 | 0.9860 | 236.9 |

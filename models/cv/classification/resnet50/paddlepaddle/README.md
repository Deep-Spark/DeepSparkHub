# ResNet50

## Model Description

ResNet50 is a deep convolutional neural network with 50 layers, known for its innovative residual learning framework. It
introduces skip connections that bypass layers, enabling the training of very deep networks by addressing vanishing
gradient problems. This architecture achieved breakthrough performance in image classification tasks, winning the 2015
ImageNet competition. ResNet50's efficient design and strong feature extraction capabilities make it widely used in
computer vision applications, serving as a backbone for various tasks like object detection and segmentation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

```bash
mkdir -p data/datasets/flowers102
cd data/datasets/flowers102
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/flowers102.tgz
tar -xf flowers102.tgz

data/datasets/flowers102
├── jpg
│   └── image_00000.jpg
│   ├── image_00001.jpg
│   └── ...
├── flowers102_label_list.txt    
├── train_extra_list.txt
└── val_list.txt
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
  - paddlepaddle-3.0.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl

```bash
mkdir -p dataset
ln -s ${DATASET_DIR}/flowers102 dataset/flowers102

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip3 install protobuf==3.20.3
pip3 install pyyaml
pip3 install -r requirements.txt
rm -rf ppcls && ln -s ppcls_2.6 ppcls
```

## Model Training

```bash
bash run_resnet50_dist.sh
```

## Model Results

| Model    | GPU        | CELoss   | loss   | top1   | top5   |
|----------|------------|----------|--------|----------|----------|
| ResNet50 | BI-V150 x8 | 4.80621  | 4.80621 | 0.05000 | 0.18529|

## Reference

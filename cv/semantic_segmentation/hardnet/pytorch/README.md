# HardNet

## Model Description

The Harmonic Densely Connected Network to achieve high efficiency in terms of both low MACs and memory traffic. The new
network achieves 35%, 36%, 30%, 32%, and 45% inference time reduction compared with FC-DenseNet-103, DenseNet-264,
ResNet-50, ResNet-152, and SSD-VGG, respectively.

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── train2017.txt
├── val2017.txt
└── ...
```

### Install Dependencies

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Model Training

```shell
bash train_hardnet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## References

- [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)
- [torchvision](../../torchvision/pytorch/README.md)

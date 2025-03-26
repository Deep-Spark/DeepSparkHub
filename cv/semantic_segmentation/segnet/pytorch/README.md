# SegNet

## Model Description

SegNet is a semantic segmentation model. This core trainable segmentation architecture consists of an encoder network, a
corresponding decoder network followed by a pixel-wise classification layer. The architecture of the encoder network is
topologically identical to the 13 convolutional layers in the VGG16 network. The role of the decoder network is to map
the low resolution encoder feature maps to full input resolution feature maps for pixel-wise classification. The novelty
of SegNet lies is in the manner in which the decoder upsamples its lower resolution input feature maps. Specifically,
the decoder uses pooling indices computed in the max-pooling step of the corresponding encoder to perform non-linear
upsampling.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

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
bash train_segnet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## References

- [torchvision](../../torchvision/pytorch/README.md)

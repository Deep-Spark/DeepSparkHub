# ICNet

## Model Description

An image cascade network (ICNet) that incorporates multi-resolution branches under proper label guidance ICNet provide
in-depth analysis of the framework and introduce the cascade feature fusion unit to quickly achieve high-quality
segmentation.

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
bash train_icnet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## References

- [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)
- [torchvision](../../torchvision/pytorch/README.md)

# DabNet

## Model Description

A novel Depthwise Asymmetric Bottleneck (DAB) module, which efficiently adopts depth-wise asymmetric convolution and dilated convolution to build a bottleneck structure.
Based on the DAB module, design a Depth-wise Asymmetric Bottleneck Network (DABNet) especially for real-time semantic segmentation.
It creates sufficient receptive field and densely utilizes the contextual information.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
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
bash train_dabnet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## References

- [SegmenTron](https://github.com/LikeLy-Journey/SegmenTron)
- [torchvision](../../torchvision/pytorch/README.md)

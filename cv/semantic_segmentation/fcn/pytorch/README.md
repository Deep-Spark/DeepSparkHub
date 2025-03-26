# FCN

## Model Description

Fully Convolutional Networks, or FCNs, are an architecture used mainly for semantic segmentation. They employ solely
locally connected layers, such as convolution, pooling and upsampling. Avoiding the use of dense layers means less
parameters (making the networks faster to train). It also means an FCN can work for variable image sizes given all
connections are local. The network consists of a downsampling path, used to extract and interpret the context, and an
upsampling path, which allows for localization. FCNs also employ skip connections to recover the fine-grained spatial
information lost in the downsampling path.

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
bash train_fcn_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## References

- [torchvision](../../torchvision/pytorch/README.md)

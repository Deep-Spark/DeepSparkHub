# DeepLabV3

## Model Description

DeepLabV3 is a semantic segmentation architecture that improves upon DeepLabV2 with several modifications. To handle the
problem of segmenting objects at multiple scales, modules are designed which employ atrous convolution in cascade or in
parallel to capture multi-scale context by adopting multiple atrous rates. 

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
bash train_deeplabv3_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## References

- [torchvision](../../torchvision/pytorch/README.md)

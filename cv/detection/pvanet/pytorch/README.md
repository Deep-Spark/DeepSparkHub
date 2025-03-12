# PVANet

## Model Description

PVANet is an efficient deep learning model for object detection, designed to minimize computational cost while
maintaining high accuracy. It employs a lightweight architecture based on the principle of "more layers with fewer
channels," incorporating innovations like C.ReLU and Inception structures. PVANet achieves competitive results on VOC
benchmarks with significantly reduced computational requirements compared to heavier networks. Its optimized design
makes it suitable for real-time applications where both speed and accuracy are crucial.

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
# Multiple GPUs on one machine
bash train_pvanet_dist.sh --data-path /path/to/coco2017/ --dataset coco

# On single GPU
python3 train.py --data-path /path/to/coco2017/ --dataset coco
```

## References

- [pytorch_imagenet](https://github.com/sanghoon/pytorch_imagenet)
- [vision](https://github.com/pytorch/vision)
# RetinaNet

## Model description

The paper proposes a method to convert a deep learning object detector into an equivalent spiking neural network. The aim is to provide a conversion framework that is not constrained to shallow network structures and classification problems as in state-of-the-art conversion libraries. The results show that models of higher complexity, such as the RetinaNet object detector, can be converted with limited loss in performance.

## Step 1: Installing packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Preparing COCO dataset

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

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

## Step 3: Training on COCO dataset

Download the [COCO Dataset](https://cocodataset.org/#home) 

### Multiple GPUs on one machine

```shell
bash train_retinanet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

### Parameters

Ref: [torchvision](../../torchvision/pytorch/README.md)

## Reference

https://github.com/pytorch/vision
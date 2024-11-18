# Mask R-CNN

## Model description

Nuclei segmentation is both an important and in some ways ideal task for modern computer vision methods, e.g. convolutional neural networks. While recent developments in theory and open-source software have made these tools easier to implement, expert knowledge is still required to choose the right model architecture and training setup. We compare two popular segmentation frameworks, U-Net and Mask-RCNN in the nuclei segmentation task and find that they have different strengths and failures. To get the best of both worlds, we develop an ensemble model to combine their predictions that can outperform both models by a significant margin and should be considered when aiming for best nuclei segmentation performance.

## Step 1: Installing packages
```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

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

```bash
mkdir -p ./datasets/
ln -s /path/to/coco2017 ./datasets/coco
```

## Step 3: Training

### Single Card
python train.py --data-path ./datasets/coco --dataset coco --model maskrcnn_resnet50_fpn --lr 0.001 --batch-size 4

### AMP 
python train.py --data-path ./datasets/coco --dataset coco --model maskrcnn_resnet50_fpn --lr 0.001 --batch-size 1 --amp

### DDP
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py\
    --data-path ./datasets/coco --dataset coco --model maskrcnn_resnet50_fpn --wd 0.000001 --lr 0.001 --batch-size 4
```
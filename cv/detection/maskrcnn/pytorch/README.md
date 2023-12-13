# Mask R-CNN

## Model description

Nuclei segmentation is both an important and in some ways ideal task for modern computer vision methods, e.g. convolutional neural networks. While recent developments in theory and open-source software have made these tools easier to implement, expert knowledge is still required to choose the right model architecture and training setup. We compare two popular segmentation frameworks, U-Net and Mask-RCNN in the nuclei segmentation task and find that they have different strengths and failures. To get the best of both worlds, we develop an ensemble model to combine their predictions that can outperform both models by a significant margin and should be considered when aiming for best nuclei segmentation performance.

## Step 1: Installing packages
```
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets
```
source download_dataset.sh
```

## Step 3: Training

```
cd pytorch
```
### Mask R-CNN on Coco using 8 cards:
```
bash run_8cards.sh
```
### Mask R-CNN on Coco using 4 cards:
```
bash run_4cards.sh
```
### Mask R-CNN on Coco using 1 cards:
```
bash run_single.sh
```

## Results on BI-V100

| GPUs | FP16  | FPS | mAP  |
|------|-------|-----|------|
| 1x8  | False | 8.5 | 0.377 |


## Reference
https://github.com/mlcommons/training

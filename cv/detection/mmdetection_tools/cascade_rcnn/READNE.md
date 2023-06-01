# Cascade R-CNN

> [Cascade R-CNN: High Quality Object Detection and Instance Segmentation](https://arxiv.org/abs/1906.09756)

<!-- [ALGORITHM] -->

## Abstract

In object detection, the intersection over union (IoU) threshold is frequently used to define positives/negatives. The threshold used to train a detector defines its quality. While the commonly used threshold of 0.5 leads to noisy (low-quality) detections, detection performance frequently degrades for larger thresholds. This paradox of high-quality detection has two causes: 1) overfitting, due to vanishing positive samples for large thresholds, and 2) inference-time quality mismatch between detector and test hypotheses. A multi-stage object detection architecture, the Cascade R-CNN, composed of a sequence of detectors trained with increasing IoU thresholds, is proposed to address these problems. The detectors are trained sequentially, using the output of a detector as training set for the next. This resampling progressively improves hypotheses quality, guaranteeing a positive training set of equivalent size for all detectors and minimizing overfitting. The same cascade is applied at inference, to eliminate quality mismatches between hypotheses and detectors. An implementation of the Cascade R-CNN without bells or whistles achieves state-of-the-art performance on the COCO dataset, and significantly improves high-quality detection on generic and specific object detection datasets, including VOC, KITTI, CityPerson, and WiderFace. Finally, the Cascade R-CNN is generalized to instance segmentation, with nontrivial improvements over the Mask R-CNN.

## Step 1: Training

### On single GPU

```bash
python3 tools/train.py  configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py 
```

### Multiple GPUs on one machine

```bash
bash tools/dist_train.sh  configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py  8
```

|       model       |     GPU     | FP32                                 | 
|-------------------| ----------- | ------------------------------------ |
|   Cascade R-CNN   | 8 cards     | MAP=40.4                             |


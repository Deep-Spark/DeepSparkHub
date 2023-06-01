# CornerNet

> [Cornernet: Detecting objects as paired keypoints](https://arxiv.org/abs/1808.01244)

<!-- [ALGORITHM] -->

## Abstract

We propose CornerNet, a new approach to object detection where we detect an object bounding box as a pair of keypoints, the top-left corner and the bottom-right corner, using a single convolution neural network. By detecting objects as paired keypoints, we eliminate the need for designing a set of anchor boxes commonly used in prior single-stage detectors. In addition to our novel formulation, we introduce corner pooling, a new type of pooling layer that helps the network better localize corners. Experiments show that CornerNet achieves a 42.2% AP on MS COCO, outperforming all existing one-stage detectors.

## Step 1: Training

### On single GPU

```bash
python3 tools/train.py configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py
```

### Multiple GPUs on one machine

```bash
bash tools/dist_train.sh configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py 8
```

|         model       |     GPU     | FP32                                 | 
|---------------------| ----------- | ------------------------------------ |
|   HourglassNet-104  | 8 cards     | MAP=41.2                             |

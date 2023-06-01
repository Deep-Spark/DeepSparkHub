# RepPoints

> [RepPoints: Point Set Representation for Object Detection](https://arxiv.org/abs/1904.11490)

<!-- [ALGORITHM] -->

## Abstract

Modern object detectors rely heavily on rectangular bounding boxes, such as anchors, proposals and the final predictions, to represent objects at various recognition stages. The bounding box is convenient to use but provides only a coarse localization of objects and leads to a correspondingly coarse extraction of object features. In this paper, we present RepPoints(representative points), a new finer representation of objects as a set of sample points useful for both localization and recognition. Given ground truth localization and recognition targets for training, RepPoints learn to automatically arrange themselves in a manner that bounds the spatial extent of an object and indicates semantically significant local areas. They furthermore do not require the use of anchors to sample a space of bounding boxes. We show that an anchor-free object detector based on RepPoints can be as effective as the state-of-the-art anchor-based detection methods, with 46.5 AP and 67.4 AP50 on the COCO test-dev detection benchmark, using ResNet-101 model.


## Step 1: Training

### On single GPU

```bash
python3 tools/train.py configs/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py
```

### Multiple GPUs on one machine

```bash
bash tools/dist_train.sh configs/reppoints/reppoints_moment_r101_fpn_dconv_c3-c5_gn-neck+head_2x_coco.py 8
```

|   model     |     GPU     | FP32                                 | 
|-------------| ----------- | ------------------------------------ |
|   RepPoints | 8 cards     | MAP=43.2                             |
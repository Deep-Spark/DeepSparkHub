# DBNet

## Model description
Recently, segmentation-based methods are quite popular in scene text detection, as the segmentation results can more accurately describe scene text of various shapes such as curve text. However, the post-processing of binarization is essential for segmentation-based detection, which converts probability maps produced by a segmentation method into bounding boxes/regions of text. In this paper, we propose a module named Differentiable Binarization (DB), which can perform the binarization process in a segmentation network. Optimized along with a DB module, a segmentation network can adaptively set the thresholds for binarization, which not only simplifies the post-processing but also enhances the performance of text detection. Based on a simple segmentation network, we validate the performance improvements of DB on five benchmark datasets, which consistently achieves state-of-the-art results, in terms of both detection accuracy and speed. In particular, with a light-weight backbone, the performance improvements by DB are significant so that we can look for an ideal tradeoff between detection accuracy and efficiency.
## Step 2: Preparing datasets

```shell
$ mkdir data
$ cd data
```
ICDAR 2015
Please [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads) download ICDAR 2015 here
ch4_training_images.zip、ch4_test_images.zip、ch4_training_localization_transcription_gt.zip、Challenge4_Test_Task1_GT.zip

```shell
mkdir icdar2015 && cd icdar2015
mkdir imgs && mkdir annotations

mv ch4_training_images imgs/training
mv ch4_test_images imgs/test

mv ch4_training_localization_transcription_gt annotations/training
mv Challenge4_Test_Task1_GT annotations/test
```
Please [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json) download instances_training.json here
Please [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json) download instances_test.json here

```shell

icdar2015/
├── imgs
│   ├── test
│   └── training
├── instances_test.json
└── instances_training.json

```
### Build Extension

```shell
$ DBNET_CV_WITH_OPS=1 python3 setup.py build && cp build/lib.linux*/dbnet_cv/_ext.cpython* dbnet_cv
```
### Install packages

```shell
$ pip3 install -r requirements.txt
```

### Training on single card
```shell
$ python3 train.py configs/textdet/dbnet/dbnet_mobilenetv3_fpnc_1200e_icdar2015.py
```

### Training on mutil-cards
```shell
$ bash dist_train.sh configs/textdet/dbnet/dbnet_mobilenetv3_fpnc_1200e_icdar2015.py 8
```

## Results on BI-V100

| approach|  GPUs   | train mem | train FPS |
| :-----: |:-------:| :-------: |:--------: |
|  dbnet  | BI100x8 |   5426    |  54.375   |

|0_hmean-iou:recall: |  0_hmean-iou:precision:  | 0_hmean-iou:hmean:|
|      :-----:       |       :-------:          |     :-------:     |
|      0.7111        |          0.8062          |       0.7557      |  

## Reference
https://github.com/open-mmlab/mmocr

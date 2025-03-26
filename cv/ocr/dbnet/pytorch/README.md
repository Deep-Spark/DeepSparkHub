# DBNet

## Model Description

Recently, segmentation-based methods are quite popular in scene text detection, as the segmentation results can more
accurately describe scene text of various shapes such as curve text. However, the post-processing of binarization is
essential for segmentation-based detection, which converts probability maps produced by a segmentation method into
bounding boxes/regions of text. In this paper, we propose a module named Differentiable Binarization (DB), which can
perform the binarization process in a segmentation network. Optimized along with a DB module, a segmentation network can
adaptively set the thresholds for binarization, which not only simplifies the post-processing but also enhances the
performance of text detection. Based on a simple segmentation network, we validate the performance improvements of DB on
five benchmark datasets, which consistently achieves state-of-the-art results, in terms of both detection accuracy and
speed. In particular, with a light-weight backbone, the performance improvements by DB are significant so that we can
look for an ideal tradeoff between detection accuracy and efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

```shell
mkdir data
cd data
```

Download [ICDAR 2015](https://rrc.cvc.uab.es/?ch=4&com=downloads).

- ch4_training_images.zip
- ch4_test_images.zip
- ch4_training_localization_transcription_gt.zip
- Challenge4_Test_Task1_GT.zip

```shell
mkdir icdar2015 && cd icdar2015
mkdir imgs && mkdir annotations

mv ch4_training_images imgs/training
mv ch4_test_images imgs/test

mv ch4_training_localization_transcription_gt annotations/training
mv Challenge4_Test_Task1_GT annotations/test
```

Download [instances_training.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_training.json).

Download [instances_test.json](https://download.openmmlab.com/mmocr/data/icdar2015/instances_test.json).

```shell
icdar2015/
├── imgs
│   ├── test
│   └── training
├── instances_test.json
└── instances_training.json
```

### Install Dependencies

```shell
# Build Extension
DBNET_CV_WITH_OPS=1 python3 setup.py build && cp build/lib.linux*/dbnet_cv/_ext.cpython* dbnet_cv

# Install packages
pip3 install -r requirements.txt
```

## Model Training

```shell
# Training on single card
python3 train.py configs/textdet/dbnet/dbnet_mobilenetv3_fpnc_1200e_icdar2015.py

# Training on mutil-cards
bash dist_train.sh configs/textdet/dbnet/dbnet_mobilenetv3_fpnc_1200e_icdar2015.py 8
```

## Model Results

| Model | GPUs       | train mem | train FPS | 0_hmean-iou:recall | 0_hmean-iou:precision | 0_hmean-iou:hmean |
|-------|------------|-----------|-----------|--------------------|-----------------------|-------------------|
| DBNet | BI-V100 x8 | 5426      | 54.375    | 0.7111             | 0.8062                | 0.7557            |

## References

- [mmocr](https://github.com/open-mmlab/mmocr)SSS

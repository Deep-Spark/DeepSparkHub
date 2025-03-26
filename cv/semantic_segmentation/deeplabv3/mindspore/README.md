# DeepLabV3

## Model Description

DeepLab is a series of image semantic segmentation models, DeepLabV3 improves significantly over previous versions. Two
keypoints of DeepLabV3: Its multi-grid atrous convolution makes it better to deal with segmenting objects at multiple
scales, and augmented ASPP makes image-level features available to capture long range information. This repository
provides a script and recipe to DeepLabV3 model and achieve state-of-the-art performance.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Pascal VOC datasets [link](https://pjreddie.com/projects/pascal-voc-dataset-mirror), and Semantic Boundaries Dataset: [link](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz)

- Download segmentation dataset.

- Prepare the training data list file. The list file saves the relative path to image and annotation pairs. Lines are like:

```shell
JPEGImages/00001.jpg SegmentationClassGray/00001.png
JPEGImages/00002.jpg SegmentationClassGray/00002.png
JPEGImages/00003.jpg SegmentationClassGray/00003.png
JPEGImages/00004.jpg SegmentationClassGray/00004.png
......
```

You can also generate the list file automatically by run script: `python3 ./src/data/get_dataset_lst.py --data_root=/PATH/TO/DATA`

- Configure and run build_data.sh to convert dataset to mindrecords. Arguments in scripts/build_data.sh:

 ```shell
 --data_root                 root path of training data
 --data_lst                  list of training data(prepared above)
 --dst_path                  where mindrecords are saved
 --num_shards                number of shards of the mindrecords
 --shuffle                   shuffle or not
 ```

Please download resnet101 from [here](https://download.mindspore.cn/model_zoo/r1.2/resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78/).

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

```bash
python3 train.py --data_file=/home/dataset/deeplabv3/vocaug_train.mindrecord0 --train_dir=./ckpt \
                 --train_epochs=200 --batch_size=32 --crop_size=513 --base_lr=0.015 --lr_type=cos \
                 --min_scale=0.5 --max_scale=2.0 --ignore_label=255 --num_classes=21 \
                 --model=deeplab_v3_s16 \
                 --ckpt_pre_trained=./resnet101_ascend_v120_imagenet2012_official_cv_bs32_acc78.ckpt \
                 --save_steps=1500 --keep_checkpoint_max=200 --device_target=GPU
```

### Evaluation

```bash
python3 eval.py --data_root=deeplabv3/ --data_lst=voc_val_lst.txt --batch_size=32 --crop_size=513 \
                --ignore_label=255 --num_classes=21 --model=deeplab_v3_s16 --scales_type=0 \
                --freeze_bn=True --device_target=GPU --ckpt_path=deeplab_v3_s16-200_45.ckpt
```

## Model Results

| GPU         | per step time | Miou   |
|-------------|---------------|--------|
| BI-V100 x1  | 1.465s        | 0.7386 |
| NV-V100s x1 | 1.716s        | 0.7453 |

## References

- [Paper](https://arxiv.org/abs/1706.05587)

# SSD

## Model Description

SSD (Single Shot MultiBox Detector) is a fast and efficient object detection model that predicts bounding boxes and
class scores in a single forward pass. It uses a set of default boxes at different scales and aspect ratios across
multiple feature maps to detect objects of various sizes. SSD combines predictions from different layers to handle
objects at different resolutions, offering a good balance between speed and accuracy for real-time detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Download [Pascal VOC Dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) and reorganize the
directory as follows:

```bash
dataset/VOCROOT/
     |->VOC2007/
     |    |->Annotations/
     |    |->ImageSets/
     |    |->...
     |->VOC2012/   # use it
     |    |->Annotations/
     |    |->ImageSets/
     |    |->...
     |->VOC2007TEST/
     |    |->Annotations/
     |    |->...
```

VOCROOT is your path of the Pascal VOC Dataset.

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

cd dataset
mkdir tfrecords
pip3 install tf_slim
python3 convert_voc_sample_tfrecords.py --dataset_directory=./ --output_directory=tfrecords --train_splits VOC2012_sample --validation_splits VOC2012_sample

cd ../
```

Download the pre-trained VGG-16 model (reduced-fc) from
[Google Drive](https://drive.google.com/drive/folders/184srhbt8_uvLKeWW_Yo8Mc5wTyc0lJT7) and put them into one sub-directory
named 'model' (we support SaverDef.V2 by default, the V1 version is also available for sake of compatibility).

## Model Training

```bash
# multi gpus
python3 train_ssd.py --batch_size 16
```

## Model Results

| Model | GPU     | acc      | fps   |
|-------|---------|----------|-------|
| SSD   | BI-V100 | 0.783513 | 3.177 |

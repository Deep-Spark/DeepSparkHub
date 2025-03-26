# RetinaFace

## Model Description

Though tremendous strides have been made in uncontrolled face detection, accurate and efficient face localisation in the
wild remains an open challenge. This paper presents a robust single-stage face detector, named RetinaFace, which
performs pixel-wise face localisation on various scales of faces by taking advantages of joint extra-supervised and
self-supervised multi-task learning. Specifically, We make contributions in the following five aspects: (1) We manually
annotate five facial landmarks on the WIDER FACE dataset and observe significant improvement in hard face detection with
the assistance of this extra supervision signal. (2) We further add a self-supervised mesh decoder branch for predicting
a pixel-wise 3D shape face information in parallel with the existing supervised branches. (3) On the WIDER FACE hard
test set, RetinaFace outperforms the state of the art average precision (AP) by 1.1% (achieving AP equal to 91.4%). (4)
On the IJB-C test set, RetinaFace enables state of the art methods (ArcFace) to improve their results in face
verification (TAR=89.59% for FAR=1e-6). (5) By employing light-weight backbone networks, RetinaFace can run real-time on
a single CPU core for a VGA-resolution image.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

1. Download the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) dataset.

2. Download annotations (face bounding boxes & five facial landmarks) from [baidu
   cloud](https://pan.baidu.com/s/1Laby0EctfuJGgGMgRRgykA) or
   [dropbox](https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0)

3. Organise the dataset directory as follows:

```Shell

  ./data/widerface/
    train/
      images/
      label.txt
    val/
      images/
      wider_val.txt
      
```

4. Download pretrained models from [google cloud](https://drive.google.com/open?id=1oZRSG0ZegbVkVwUd8wUIQx8W7yfZ_ki1)
   and [baidu cloud](https://pan.baidu.com/s/12h97Fy1RYuqMMIV-RpzdPg) Password: fstq . The model could be put as
   follows:

```Shell

  ./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth

```

## Model Training

```shell
# Single GPU with mobilenet backbone
python3 train.py --network mobile0.25

# Multi GPU with resnet50 backbone
python3 train.py --network resnet50

# Evaluate
## 1. Generate txt file
python3 test_widerface.py --trained_model ${weight_file} --network mobile0.25 or resnet50

## 2. Evaluate txt results. Demo come from [Here](https://github.com/wondervictor/WiderFace-Evaluation)
cd ./widerface_evaluate
python3 setup.py build_ext --inplace
python3 evaluation.py
```

You can also use widerface official Matlab evaluate demo in
   [Here](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html)

## Model Results

| Model      | GPU     | Type       | AP                 |
|------------|---------|------------|--------------------|
| RetinaFace | BI-V100 | Easy   Val | 0.940523547724767  |
| RetinaFace | BI-V100 | Medium Val | 0.9315975713948582 |
| RetinaFace | BI-V100 | Hard   Val | 0.8303531408987751 |

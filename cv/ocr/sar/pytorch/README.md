# SAR

## Model description

Recognizing irregular text in natural scene images is challenging due to the large variance in text appearance, such as curvature, orientation and distortion. Most existing approaches rely heavily on sophisticated model designs and/or extra fine-grained annotations, which, to some extent, increase the difficulty in algorithm implementation and data collection. In this work, we propose an easy-to-implement strong baseline for irregular scene text recognition, using off-the-shelf neural network components and only word-level annotations. It is composed of a 31-layer ResNet, an LSTM-based encoder-decoder framework and a 2-dimensional attention module. Despite its simplicity, the proposed method is robust and achieves state-of-the-art performance on both regular and irregular scene text recognition benchmarks.

## Step 1: Installing packages

```shell
cd csrc/
bash clean.sh
bash build.sh
bash install.sh
cd ..
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```shell
mkdir data
ln -s /path/to/mixture ./data/
```

Download datasets from this [page](https://mmocr.readthedocs.io/zh_CN/latest/datasets/recog.html),
data folder would be like below:

```
├── mixture
│   ├── coco_text
│   │   ├── train_label.txt
│   │   ├── train_words
│   ├── icdar_2011
│   │   ├── training_label.txt
│   │   ├── Challenge1_Training_Task3_Images_GT
│   ├── icdar_2013
│   │   ├── train_label.txt
│   │   ├── test_label_1015.txt
│   │   ├── test_label_1095.txt
│   │   ├── Challenge2_Training_Task3_Images_GT
│   │   ├── Challenge2_Test_Task3_Images
│   ├── icdar_2015
│   │   ├── train_label.txt
│   │   ├── test_label.txt
│   │   ├── ch4_training_word_images_gt
│   │   ├── ch4_test_word_images_gt
│   ├── III5K
│   │   ├── train_label.txt
│   │   ├── test_label.txt
│   │   ├── train
│   │   ├── test
│   ├── ct80
│   │   ├── test_label.txt
│   │   ├── image
│   ├── svt
│   │   ├── test_label.txt
│   │   ├── image
│   ├── svtp
│   │   ├── test_label.txt
│   │   ├── image
│   ├── Syn90k
│   │   ├── shuffle_labels.txt
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── mnt
│   ├── SynthText
│   │   ├── alphanumeric_labels.txt
│   │   ├── shuffle_labels.txt
│   │   ├── instances_train.txt
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── synthtext
│   ├── SynthAdd
│   │   ├── label.txt
│   │   ├── label.lmdb
│   │   ├── SynthText_Add
│   ├── TextOCR
│   │   ├── image
│   │   ├── train_label.txt
│   │   ├── val_label.txt
│   ├── Totaltext
│   │   ├── imgs
│   │   ├── annotations
│   │   ├── train_label.txt
│   │   ├── test_label.txt
│   ├── OpenVINO
│   │   ├── image_1
│   │   ├── image_2
│   │   ├── image_5
│   │   ├── image_f
│   │   ├── image_val
│   │   ├── train_1_label.txt
│   │   ├── train_2_label.txt
│   │   ├── train_5_label.txt
│   │   ├── train_f_label.txt
│   │   ├── val_label.txt
```

## Step 3: Training

### Training on single card
```shell
python3 train.py configs/sar_r31_parallel_decoder_academic.py
```

### Training on mutil-cards
```shell
bash dist_train.sh configs/sar_r31_parallel_decoder_academic.py 8
```

## Reference
https://github.com/open-mmlab/mmocr


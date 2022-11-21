# SATRN(Self-Attention Text Recognition Network)

## Model description

Scene text recognition (STR) is the task of recognizing character sequences in natural scenes. While there have been great advances in STR methods, current methods still fail to recognize texts in arbitrary shapes, such as heavily curved or rotated texts, which are abundant in daily life (e.g. restaurant signs, product labels, company logos, etc). This paper introduces a novel architecture to recognizing texts of arbitrary shapes, named Self-Attention Text Recognition Network (SATRN), which is inspired by the Transformer. SATRN utilizes the self-attention mechanism to describe two-dimensional (2D) spatial dependencies of characters in a scene text image. Exploiting the full-graph propagation of self-attention, SATRN can recognize texts with arbitrary arrangements and large inter-character spacing. As a result, SATRN outperforms existing STR models by a large margin of 5.7 pp on average in "irregular text" benchmarks. We provide empirical analyses that illustrate the inner mechanisms and the extent to which the model is applicable (e.g. rotated and multi-line text). We will open-source the code.


## Step 1: Installing packages

```
$ cd /satrn/pytorch/base/csrc
$ bash clean.sh
$ bash build.sh
$ bash install.sh
$ cd ..
$ pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```shell
$ mkdir data
$ cd data
```
https://mmocr.readthedocs.io/zh_CN/latest/datasets/recog.html


- when done data folder looks like
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
$ python3 train.py configs/models/satrn_academic.py
```

### Training on mutil-cards
```shell
$ bash dist_train.sh configs/models/satrn_academic.py 8
```

## Results on BI-V100

| approach|  GPUs   | train mem | train FPS |
| :-----: |:-------:| :-------: |:--------: |
|  satrn  | BI100x8 |   14.159G |  549.94   |

| dataset |   acc   |
| :-----: |:-------:|
|  IIIT5K |   94.5  |
|  IC15   |   83.3  |
|  SVTP   |   88.4  |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 0.841                | SDK V2.2,bs:128,8x,fp32                  | 630         | N/A      | 166\*8     | 0.98        | 28.5\*8                 | 1         |


## Reference
https://github.com/open-mmlab/mmocr




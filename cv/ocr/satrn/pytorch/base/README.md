# SATRN

## Model Description

SATRN (Self-Attention Text Recognition Network) is an advanced deep learning model for scene text recognition,
particularly effective for texts with arbitrary shapes like curved or rotated characters. Inspired by Transformer
architecture, SATRN utilizes self-attention mechanisms to capture 2D spatial dependencies between characters. This
enables it to handle complex text arrangements and large inter-character spacing with high accuracy. SATRN significantly
outperforms traditional methods in recognizing irregular texts, making it valuable for real-world applications like sign
and logo recognition.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

```bash
mkdir data
cd data
```

Reffering to [MMOCR Docs](https://mmocr.readthedocs.io/zh_CN/dev-1.x/user_guides/data_prepare/datasetzoo.html) to
prepare datasets. Datasets path would look like below:

```bash
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

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-dev

cd /satrn/pytorch/base/csrc
bash clean.sh
bash build.sh
bash install.sh
cd ..
pip3 install -r requirements.txt
```

## Model Training

```bash
# Training on single card
python3 train.py configs/models/satrn_academic.py

# Training on mutil-cards
bash dist_train.sh configs/models/satrn_academic.py 8
```

## Model Results

| Model | GPU        | train mem | train FPS | ACC                                  |
|-------|------------|-----------|-----------|--------------------------------------|
| SATRN | BI-V100 x8 | 14.159G   | 549.94    | IIIT5K: 94.5, IC15: 83.3, SVTP: 88.4 |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 0.841                | SDK V2.2,bs:128,8x,fp32                  | 630         | 88.4     | 166\*8     | 0.98        | 28.5\*8                 | 1         |

## References

- [mmocr](https://github.com/open-mmlab/mmocr)

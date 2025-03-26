# OC-SORT

## Model Description

OC-SORT (Observation-Centric SORT) is an advanced multi-object tracking algorithm that enhances traditional SORT by
addressing limitations in Kalman filters and non-linear motion scenarios. It improves tracking robustness in crowded
scenes and complex motion patterns while maintaining simplicity and real-time performance. OC_SORT focuses on
observation-centric updates, making it more reliable for object tracking in challenging environments. It remains
flexible for integration with various detectors and matching modules, offering improved accuracy without compromising
speed.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

- **MOT17_ch datasets**

```bash
cd dataset/mot
git clone https://github.com/ifzhang/ByteTrack.git
```

Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/),
[CrowdHuman](https://www.crowdhuman.org/),
[Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md),
[ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them under
<ByteTrack_HOME>/datasets in the following structure:

```bash
datasets/
├── crowdhuman
│   ├── CrowdHuman_train01.zip
│   ├── CrowdHuman_train02.zip
│   ├── CrowdHuman_train03.zip
│   ├── CrowdHuman_val.zip
│   ├── annotation_train.odgt
│   ├── annotation_val.odgt
├── data_path
│   ├── citypersons.train
│   └── eth.train
├── mot
│   └── MOT17.zip
└── prepare_datasets.sh
```

Unzip and organize the path following below steps.

```bash
# MOT17
cd mot
unzip MOT17.zip
mv MOT17/images/train .
mv MOT17/images/test .

# CrowdHuman_train
cd ../crowdhuman
unzip CrowdHuman_train01.zip -d CrowdHuman_train
unzip CrowdHuman_train02.zip -d CrowdHuman_train
unzip CrowdHuman_train03.zip -d CrowdHuman_train
unzip CrowdHuman_val.zip -d CrowdHuman_val
mv CrowdHuman_train/Images/* CrowdHuman_train
mv CrowdHuman_val/Images/* CrowdHuman_val
```

The datasets path would be look like below.

```bash
datasets
   ├── mot
   │   └── train
   │   └── test
   └── crowdhuman
       ├── CrowdHuman_train
       ├── CrowdHuman_train
       ├── CrowdHuman_val
       ├── annotation_train.odgt
       └── annotation_val.odgt
```

Then, you need to turn the datasets to COCO format and mix different training data:

```bash
cd <ByteTrack_HOME>
python3 tools/convert_mot17_to_coco.py
python3 tools/convert_crowdhuman_to_coco.py
```

Mixing different datasets:

```bash
cd datasets
mkdir -p mix_mot_ch/annotations
cp mot/annotations/val_half.json mix_mot_ch/annotations/val_half.json
cp mot/annotations/test.json mix_mot_ch/annotations/test.json
cd mix_mot_ch
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val

cd <ByteTrack_HOME>
python3 tools/mix_data_ablation.py
```

Create a data link:

```bash
cd PaddleDetection/dataset/mot
ln -s ByteTrack/datasets/mix_mot_ch mix_mot_ch
```

- **MOT-17 half train datasets**

Download [MOT17](https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip) and put them under <PaddleDetection_HOME>/datasets/mot/ in the following structure:

```bash
MOT17/
└──images
   ├── test
   │   ├── MOT17-01-DPM
   │   │   ├── det
   │   │   └── img1
   │   ├── ...
   │   └── MOT17-14-SDP
   │       ├── det
   │       └── img1
   └── train
       ├── MOT17-02-DPM
       │   ├── det
       │   ├── gt
       │   └── img1
       ├── ...
       └── MOT17-13-SDP
           ├── det
           ├── gt
           └── img1
```

Turn the datasets to COCO format:

```bash
cd <ByteTrack_HOME>
python3 tools/convert_mot17_to_coco.py
```

Create a data link:

```bash
cd <PaddleDetection_HOME>/datasets/mot/MOT17
ln -s ../ByteTrack/datasets/mot/annotations ./
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/PaddlePaddle/PaddleDetection.git -b release2.6 --depth=1
cd PaddleDetection/

pip3 install -r requirements.txt
pip3 install protobuf==3.20.1
pip3 install urllib3==1.26.6
pip3 install scikit-learn
```

## Model Training

```bash
# mix_mot_ch datasets
python3 -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_mot_ch.yml --eval --amp

# MOT-17 half dataset training
python3 -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp

# MOT-17 half dataset evaluation on tracking
CUDA_VISIBLE_DEVICES=0 python3 tools/eval_mot.py -c configs/mot/ocsort/ocsort_ppyoloe.yml --scaled=True
```

## Model Results

| Model   | GPU        | DATASET           | IPS    | MOTA |
|---------|------------|-------------------|--------|------|
| OC-SORT | BI-V100 x8 | MOT-17 half train | 6.5907 | 57.5 |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

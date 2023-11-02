# OC_SORT

## Model description
Observation-Centric SORT (OC-SORT) is a pure motion-model-based multi-object tracker. It aims to improve tracking robustness in crowded scenes and when objects are in non-linear motion. It is designed by recognizing and fixing limitations in Kalman filter and SORT. It is flexible to integrate with different detectors and matching modules, such as appearance similarity. It remains, Simple, Online and Real-time.

## Step 1: Installation
```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

```bash
cd PaddleDetection
yum install mesa-libGL -y

pip3 install -r requirements.txt
pip3 install protobuf==3.20.1
pip3 install urllib3==1.26.6
pip3 install scikit-learn
```

## Step 2: Prepare datasets

- MOT17_ch datasets
```bash
cd dataset/mot
git clone https://github.com/ifzhang/ByteTrack.git
```

Download [MOT17](https://motchallenge.net/), [MOT20](https://motchallenge.net/), [CrowdHuman](https://www.crowdhuman.org/), [Cityperson](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md), [ETHZ](https://github.com/Zhongdao/Towards-Realtime-MOT/blob/master/DATASET_ZOO.md) and put them under <ByteTrack_HOME>/datasets in the following structure:

```bash
datasets
   |——————mot
   |        └——————train
   |        └——————test
   └——————crowdhuman
             └——————CrowdHuman_train
             └——————CrowdHuman_val
             └——————annotation_train.odgt
             └——————annotation_val.odgt

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

- MOT-17 half train datasets
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

## Step 3: Training

```bash
cd PaddleDetection

# mix_mot_ch datasets
python3 -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/yolox_x_24e_800x1440_mix_mot_ch.yml --eval --amp

# MOT-17 half dataset training
python3 -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp

# MOT-17 half dataset evaluation
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml

```

## Results

| GPUs        | DATASET   | IPS       | MOTA     |
|-------------|-----------|-----------|----------|
| BI-V100 x8  | MOT-17 half train| 6.5907 | 57.5 | 

## Reference
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

# ByteTrack

## Model Description

ByteTrack is an efficient multi-object tracking (MOT) model that improves tracking accuracy by associating every
detection box, including low-score ones, rather than discarding them. It addresses challenges like occluded objects and
fragmented trajectories by leveraging similarities between detections and tracklets. ByteTrack achieves state-of-the-art
performance on benchmarks like MOT17, with high MOTA, IDF1, and HOTA scores while maintaining real-time processing
speeds. Its simple yet effective design makes it a robust solution for various object tracking applications in video
analysis.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

Go to visit [MOT17 official website](https://motchallenge.net/), then download the MOT17 dataset, or you can download
via [paddledet data](https://bj.bcebos.com/v1/paddledet/data/mot/MOT17.zip),then extract and place it in the
dataset/mot/folder.

The dataset path structure sholud look like:

```bash
datasets/mot/MOT17/
├── annotations
│   ├── train_half.json
│   └── train.json
|   └── val_half.json
└── images
│   ├── train
│   └── half
│   └── test
└── labels_with_ids
│   ├── train

```

### Install Dependencies

```bash
git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 
pip3 install urllib3==1.26.6
yum install mesa-libGL
python3 setup.py develop
```

## Model Training

```bash
# One GPU
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp

# Eight GPUs
python3 -m paddle.distributed.launch --log_dir=ppyoloe --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/mot/bytetrack/detector/ppyoloe_crn_l_36e_640x640_mot17half.yml --eval --amp
```

## Model Results

| Model     | GPU        | FPS    | mAP(0.5:0.95) |
|-----------|------------|--------|---------------|
| ByteTrack | BI-V100 x8 | 4.6504 | 0.538         |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

# YOLOv5

## Model description

YOLOv5 is the world's most loved vision AI, representing <a href="https://ultralytics.com">Ultralytics</a> open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

## Step 1: Installation

```bash
cd deepsparkhub/cv/detection/yolov5/paddlepaddle/
git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleYOLO.git
cd PaddleYOLO/
pip3 install -r requirements.txt
python3 setup.py develop
```

## Step 2: Preparing datasets

```bash
python3 dataset/coco/download_coco.py
```

## Step 3: Training

```bash
# Make sure your dataset path is the same as above
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Select $config_yaml from "configs/yolov5" as you wish
config_yaml=configs/yolov5/yolov5_s_300e_coco.yml
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config_yaml} --amp --eval
```

## Results

GPUs|Model|FPS|ACC
----|---|---|---
BI-V100 x8|YOLOv5-n| 10.9788 images/s | bbox ap: 0.259

## Reference

- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)

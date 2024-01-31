# YOLOv5

## Model description

WOLOv5 🚀 is the world's most loved vision AI, representing <a href="https://ultralytics.com">Ultralytics</a> open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.

We hope that the resources here will help you get the most out of YOLOv5. Please browse the YOLOv5 <a href="https://docs.ultralytics.com/yolov5">Docs</a> for details, raise an issue on <a href="https://github.com/ultralytics/yolov5/issues/new/choose">GitHub</a> for support, and join our <a href="https://ultralytics.com/discord">Discord</a> community for questions and discussions!

To request an Enterprise License please complete the form at [Ultralytics Licensing](https://ultralytics.com/license).

## Step 1: Installing

```bash
git clone --recursive -b release/2.6 https://github.com/PaddlePaddle/PaddleYOLO.git
cd PaddleYOLO
pip3 install -r requirements.txt  # install
```

## Step 2: Download data

```bash
python3 dataset/coco/download_coco.py
```

## Step 3: Run YOLOv5

```bash
# Make sure your dataset path is the same as above
cd PaddleDetection
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/yolov5/yolov5_s_300e_coco.yml --eval
```

## Reference
- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)
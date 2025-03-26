# YOLOv5

## Model Description

YOLOv5 is a state-of-the-art object detection model that builds upon the YOLO architecture, offering improved speed and
accuracy. It features a streamlined design with enhanced data augmentation and anchor box strategies. YOLOv5 supports
multiple model sizes (n/s/m/l/x) for different performance needs. The model is known for its ease of use, fast training,
and efficient inference, making it popular for real-time detection tasks across various applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone -b release/2.7 --depth=1 https://github.com/PaddlePaddle/PaddleYOLO.git
cd PaddleYOLO/

python3 dataset/coco/download_coco.py
or
ln -s /path/to/coco2017 dataset/coco
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
python3 setup.py develop
```

## Model Training

```bash
# Make sure your dataset path is the same as above
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Select $config_yaml from "configs/yolov5" as you wish
config_yaml=configs/yolov5/yolov5_s_300e_coco.yml
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config_yaml} --amp --eval
```

## Model Results

| Model    | GPU        | FPS              | ACC            |
|----------|------------|------------------|----------------|
| YOLOv5-n | BI-V100 x8 | 10.9788 images/s | bbox ap: 0.259 |

## References

- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)

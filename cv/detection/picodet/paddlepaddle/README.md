# PP-PicoDet

## Model Description

PP-PicoDet is an ultra-lightweight real-time object detection model designed for efficient deployment. It comes in four
sizes (XS/S/M/L) and incorporates advanced structures like TAL, ETA Head, and PAN to boost accuracy. The model supports
end-to-end inference by including post-processing in the network, enabling direct prediction outputs. PP-PicoDet
achieves an excellent balance between speed and accuracy, making it ideal for applications requiring real-time detection
on resource-constrained devices.

## Model Preparation

### Prepare Resources

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/PaddlePaddle/PaddleDetection.git -b release2.6 --depth=1

cd PaddleDetection/
# Get COCO Dataset
python3 dataset/coco/download_coco.py
```

### Install Dependencies

```bash
pip install -r requirements.txt
python3 setup.py install

pip3 install protobuf==3.20.3
pip3 install numba==0.56.4

yum install mesa-libGl -y
```

## Model Training

Assuming we are going to train picodet-l, the model config file is 'configs/picodet/picodet_l_640_coco_lcnet.yml' vim
configs/datasets/coco_detection.yml, set 'dataset_dir' in the configuration file to coco2017, then start trainging.

```bash
# Single GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/picodet/picodet_l_640_coco_lcnet.yml --eval

# Multi GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/picodet/picodet_l_640_coco_lcnet.yml --eval
```

## Model Results

| Model      | GPU        | IPS   | mAP0.5:0.95 | mAP0.5 |
|------------|------------|-------|-------------|--------|
| PP-PicoDet | BI-V100 x8 | 19.84 | 41.2        | 58.2   |

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/picodet/README_en.md)

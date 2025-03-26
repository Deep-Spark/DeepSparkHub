# PP-YOLOE+

## Model Description

PP-YOLOE+ is an enhanced version of PP-YOLOE, a high-performance anchor-free object detection model. It builds upon
PP-YOLOv2's architecture, offering improved accuracy and efficiency. The model comes in multiple sizes (s/m/l/x)
configurable through width and depth multipliers. PP-YOLOE+ maintains hardware compatibility by avoiding special
operators while achieving state-of-the-art speed-accuracy trade-offs. Its optimized architecture makes it ideal for
real-time applications, offering superior detection performance across various scenarios and hardware platforms.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleYOLO.git
cd PaddleYOLO/

python3 dataset/coco/download_coco.py
or
ln -s /path/to/coco2017 dataset/coco
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

> HINT:
> --eval : training with evaluation
> --amp  : Mixed-precision training

```bash
model_name=ppyoloe
job_name=ppyoloe_plus_crn_s_80e_coco
config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
weights=output/${job_name}/model_final.pdparams
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams

# Single card training
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c ${config} --eval --amp

# Multi cards training
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# Evaluation (--classwise: output mAP of each class)
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c ${config} -o weights=${weights} --classwise

# Inference (single picture or pictures path)
CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5
# CUDA_VISIBLE_DEVICES=0 python3 tools/infer.py -c ${config} -o weights=${weights} --infer_dir=demo/ --draw_threshold=0.5
```

## Model Results

| Model     | GPU        | FPS        | ACC                      |
|-----------|------------|------------|--------------------------|
| PP-YOLOE+ | BI-V100 x8 | ips:6.3293 | Best test bbox ap: 0.528 |

## References

- [Paper](https://arxiv.org/pdf/2203.16250v3.pdf)
- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)

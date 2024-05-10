# PP-YOLOE+

## Model description

PP-YOLOE is an excellent single-stage anchor-free model based on PP-YOLOv2, surpassing a variety of popular YOLO models. PP-YOLOE has a series of models, named s/m/l/x, which are configured through width multiplier and depth multiplier. PP-YOLOE avoids using special operators, such as Deformable Convolution or Matrix NMS, to be deployed friendly on various hardware.

## Step 1: Installation

```bash
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleYOLO.git
cd PaddleYOLO/
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```bash
python3 dataset/coco/download_coco.py
```

## Step 3: Training

> **HINT:**
> 
> --eval : training with evaluation
>
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

## Results


| GPUs       | FPS        | ACC                      |
| ------------ | ------------ | -------------------------- |
| BI-V100 x8 | ips:6.3293 | Best test bbox ap: 0.528 |

## Reference

- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)
- [PP-YOLOE](https://arxiv.org/pdf/2203.16250v3.pdf)

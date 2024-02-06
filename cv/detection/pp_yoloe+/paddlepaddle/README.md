# PP-YOLOE+

## Model description
- Pre training model using large-scale data set obj365
- In the backbone, add the alpha parameter to the block branch
- Optimize the end-to-end inference speed and improve the training convergence speed

## Step 1: Installing
```
git clone -b develop https://github.com/PaddlePaddle/PaddleYOLO.git
```

```
cd PaddleYOLO
pip3 install -r requirements.txt
```

## Step 2: Prepare Datasets
```
python3 dataset/coco/download_coco.py
```

## Step 3: Training

```
cd PaddleYOLO

model_name=ppyoloe # 可修改，如 yolov7
job_name=ppyoloe_plus_crn_s_80e_coco # 可修改，如 yolov7_tiny_300e_coco

config=configs/${model_name}/${job_name}.yml
log_dir=log_dir/${job_name}
# weights=https://bj.bcebos.com/v1/paddledet/models/${job_name}.pdparams
weights=output/${job_name}/model_final.pdparams

# 1.训练（单卡/多卡），加 --eval 表示边训边评估，加 --amp 表示混合精度训练
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c ${config} --eval --amp
# python3 -m paddle.distributed.launch --log_dir=${log_dir} --gpus 0,1,2,3,4,5,6,7 tools/train.py -c ${config} --eval --amp

# 2.评估，加 --classwise 表示输出每一类mAP
# CUDA_VISIBLE_DEVICES=0 python tools/eval.py -c ${config} -o weights=${weights} --classwise

# 3.预测 (单张图/图片文件夹）
# CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_img=demo/000000014439_640x640.jpg --draw_threshold=0.5
# CUDA_VISIBLE_DEVICES=0 python tools/infer.py -c ${config} -o weights=${weights} --infer_dir=demo/ --draw_threshold=0.5
```

## Reference
- [PaddleYOLO](https://github.com/PaddlePaddle/PaddleYOLO)
- [PP-YOLOE](https://arxiv.org/pdf/2203.16250v3.pdf)

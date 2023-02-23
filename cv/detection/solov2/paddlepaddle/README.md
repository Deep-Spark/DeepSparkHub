# SOLOv2

## Model description
SOLOv2 (Segmenting Objects by Locations) is a fast instance segmentation framework with strong performance. 

## 克隆代码

```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

## 安装PaddleDetection

```
cd PaddleDetection
pip install -r requirements.txt
python3 setup.py install
```

## 下载COCO数据集

```
python3 dataset/coco/download_coco.py
```

## 运行代码

```
# GPU多卡训练
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml --eval

# GPU单卡训练
export CUDA_VISIBLE_DEVICES=0

python3 tools/train.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml --eval

# 注：默认学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整config中的学习率（例如，除以8）

```

## Results on BI-V100

| GPUs | FPS | Train Epochs | mAP  |
|------|-----|--------------|------|
| 1x8  |  | 12           |  |
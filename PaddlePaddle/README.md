# 克隆代码

```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

# 安装PaddlePaddle

```
# CUDA10.2
python3 -m pip install paddlepaddle-gpu==2.2.2 -i https://mirror.baidu.com/pypi/simple
```

# 安装PaddleDetection

```
cd PaddleDetection
pip install -r requirements.txt
python3 setup.py install
```

# 下载COCO数据集

```
python3 dataset/coco/download_coco.py
```

# 运行代码

```
# GPU多卡训练示例
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# RetinaNet
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml

# FCOS
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/fcos/fcos_r50_fpn_1x_coco.yml

# DETR
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/detr/detr_r50_1x_coco.yml

# CenterNet
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml

# SOLOv2
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml


# GPU单卡训练
export CUDA_VISIBLE_DEVICES=0

# RetinaNet
python3 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml

# FCOS
python3 tools/train.py -c configs/fcos/fcos_r50_fpn_1x_coco.yml

# DETR
python3 tools/train.py -c configs/detr/detr_r50_1x_coco.yml

# CenterNet
python3 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml

# SOLOv2
python3 tools/train.py -c configs/solov2/solov2_r50_fpn_1x_coco.yml

# 注：默认学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整config中的学习率（例如，除以8）
```
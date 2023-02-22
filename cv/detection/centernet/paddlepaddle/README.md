# CenterNet

## Model description
Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

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
# GPU多卡训练示例
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

训练
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml

边训练边评估
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml --eval

# GPU单卡训练
export CUDA_VISIBLE_DEVICES=0

训练
python3 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml

边训练边评估
python3 tools/train.py -c configs/centernet/centernet_r50_140e_coco.yml --eval

# 注：默认学习率是适配多GPU训练(8x GPU)，若使用单GPU训练，须对应调整config中的学习率（例如，除以8）

```
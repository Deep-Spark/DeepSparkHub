# PP-YOLOE

## Model description
PP-YOLOE is an excellent single-stage anchor-free model based on PP-YOLOv2, surpassing a variety of popular YOLO models. PP-YOLOE has a series of models, named s/m/l/x, which are configured through width multiplier and depth multiplier. PP-YOLOE avoids using special operators, such as Deformable Convolution or Matrix NMS, to be deployed friendly on various hardware.

## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

```
cd PaddleDetection
pip3 install -r requirements.txt
```

## Step 2: Prepare Datasets
```
python3 dataset/coco/download_coco.py
```

## Step 3: Training

```
cd PaddleDetection
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml --use_vdl=true --eval -o log_iter=5
```

## Reference
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

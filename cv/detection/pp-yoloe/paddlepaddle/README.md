# PP-YOLOE

## Model Description

PP-YOLOE is a high-performance single-stage anchor-free object detection model built upon PP-YOLOv2. It outperforms
various popular YOLO variants while maintaining deployment-friendly characteristics. The model comes in multiple sizes
(s/m/l/x) configurable through width and depth multipliers. PP-YOLOE avoids special operators like Deformable
Convolution, ensuring compatibility with diverse hardware. It achieves excellent speed-accuracy trade-offs, making it
suitable for real-time applications. The model's efficient architecture and optimization techniques make it a top choice
for object detection tasks.

## Model Preparation

### Prepare Resources

```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection/
# Get COCO Dataset
python3 dataset/coco/download_coco.py
```

### Install Dependencies

```bash
pip install -r requirements.txt
python3 setup.py install
```

## Model Training

```bash
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 \
    tools/train.py \
    -c configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml \
    --use_vdl=true \
    --eval \
    -o log_iter=5
```

## References

- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

# MODEL_NAME

## Model description
The paper proposes a method to convert a deep learning object detector into an equivalent spiking neural network. The aim is to provide a conversion framework that is not constrained to shallow network structures and classification problems as in state-of-the-art conversion libraries. The results show that models of higher complexity, such as the RetinaNet object detector, can be converted with limited loss in performance.

## Step 1: Installation

```bash
pip install -r requirements.txt
python3 setup.py install
```

## Step 2: Preparing datasets

```bash
python3 dataset/coco/download_coco.py
```

## Step 3: Training

```bash
# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval

# 1 GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/retinanet/retinanet_r50_fpn_1x_coco.yml --eval
```

## Results

| GPUs        | FPS       | ACC          |
|-------------|-----------|--------------|
| BI-V100 x 8 | 251       | 71.2         |

## Reference

- [Reference_link](https://github.com/reference_repo/reference_repo)
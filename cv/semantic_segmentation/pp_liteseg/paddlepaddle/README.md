# PP-LiteSeg

## Model description

PP-LiteSeg is a novel lightweight model for the real-time semantic segmentation task. Specifically, the model presents a Flexible and Lightweight Decoder (FLD) to reduce computation overhead of previous decoder. To strengthen feature representations, this model proposes a Unified Attention Fusion Module (UAFM), which takes advantage of spatial and channel attention to produce a weight and then fuses the input features with the weight. Moreover, a Simple Pyramid Pooling Module (SPPM) is proposed to aggregate global context with low computation cost.

## Step 1: Installation

```
git clone -b release/2.8 https://github.com/PaddlePaddle/PaddleSeg.git

cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3
yum install mesa-libGL 
pip3 install paddleseg

```


## Step 2: Preparing datasets

```
mkdir -p data && cd data
wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar
wget https://paddleseg.bj.bcebos.com/dataset/camvid.tar

tar -xvf cityscapes.tar
tar -xvf camvid.tar

rm -rf cityscapes.tar
rm -rf camvid.tar
```

the unzipped dataset structure sholud look like:

```
PaddleSeg/data
├── cityscapes
│   ├── gtFine
│   ├── infer.list
│   ├── leftImg8bit
│   ├── test.list
│   ├── train.list
│   ├── trainval.list
│   └── val.list
├── camvid
│   ├── annot
│   ├── images
│   ├── README.md
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
```

## Step 3: Training

```
cd ..

# 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k

python3 -m paddle.distributed.launch tools/train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --save_interval 4000 \
    --do_eval \
    --use_vdl

# 1 GPU
export CUDA_VISIBLE_DEVICES=0
export model=pp_liteseg_stdc1_cityscapes_1024x512_scale0.5_160k

python3 tools/train.py \
    --config configs/pp_liteseg/${model}.yml \
    --save_dir output/${model} \
    --save_interval 4000 \
    --do_eval \
    --use_vdl
```

## Results

| Method | Backbone | Training Iters | FPS (BI x 8)  | mIOU |
| ------ | --------- | ------ | --------  |--------------:|
|  PP-LiteSeg-T | STDC1  |  160000  |   28.8    | 73.19% |

## Reference
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
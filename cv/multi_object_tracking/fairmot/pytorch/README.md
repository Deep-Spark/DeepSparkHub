# FairMOT

## Model Description

FairMOT is an innovative multi-object tracking model that unifies detection and re-identification in a single framework.
It features two homogeneous branches: one for anchor-free object detection and another for re-ID feature extraction.
Operating on high-resolution feature maps, FairMOT achieves fairness between detection and re-ID tasks, resulting in
improved tracking accuracy. Its joint learning approach eliminates the need for cascaded processing, making it more
efficient and effective for complex tracking scenarios in crowded environments.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Download MOT17 dataset

- [Baidu NetDisk](https://pan.baidu.com/s/1lHa6UagcosRBz-_Y308GvQ)
- [Google Drive](https://drive.google.com/file/d/1ET-6w12yHNo8DKevOVgK1dBlYs739e_3/view?usp=sharing)
- [Original dataset webpage: MOT-17](https://motchallenge.net/data/MOT17/)

```shell
# Download MOT17
mkdir -p data/MOT
cd data/MOT

```

```shell
unzip -q MOT17.zip
mkdir MOT17/images && mkdir MOT17/labels_with_ids
mv ./MOT17/train ./MOT17/images/ && mv ./MOT17/test ./MOT17/images/

cd ../../
python3 src/gen_labels_17.py

## The dataset path looks like below
data/
└── MOT
    └── MOT17
        ├── images
        │   ├── test
        │   └── train
        └── labels_with_ids
            └── train
```

Download Pretrained models

- DLA-34 COCO pretrained model: [DLA-34 official](https://drive.google.com/file/d/18Q3fzzAsha_3Qid6mn4jcIFPeOGUaj1d)
- HRNetV2-W18 ImageNet pretrained model: [BaiduYun（Access Code: r5xn)](https://pan.baidu.com/s/1Px_g1E2BLVRkKC5t-b-R5Q)
- HRNetV2-W18 ImageNet pretrained model: [BaiduYun（Access Code: itc1)](https://pan.baidu.com/s/1xn92PSCg5KtXkKcnnLOycw)

```shell
# Download ctdet_coco_dla_2x
mkdir -p models
cd models
```

### Install Dependencies

```shell
pip3 install -r requirements.txt
pip3 install pandas progress
```

## Model Training

The available train scripts are as follows:

```shell
train_dla34_mot17.sh
train_hrnet18_mot17.sh
train_hrnet32_mot17.sh

# On single GPU
GPU_NUMS=1 bash train_dla34_mot17.sh --gpus 0 --batch_size 18

# Multiple GPUs on one machine
GPU_NUMS=8 bash train_dla34_mot17.sh --gpus 0,1,2,3,4,5,6,7 --batch_size 144
```

To reduce training time span, you can append "--num_epochs 1 --num_iters 300" to the command.

Evaluate on tesing datasets.

> Only command on testing dataset is provided. MOT dataset is not public. To access MOT, submit via
> [motchallenge](https://motchallenge.net/instructions/).

```shell
cd /path/to/modelzoo/official/cv/tracking/fairmot/pytorch/src
python3 track.py mot --val_mot17 True --load_model /path/to/saved/model --conf_thres 0.4 --data_dir ../data/MOT
```

## Model Results

| Model   | GPU        | FPS  | MOTA |
|---------|------------|------|------|
| FairMOT | BI-V100 x8 | 28.5 | 69.8 |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| MOTA:69.8            | SDK V2.2,bs:64,8x,fp32                   | 52          | 69.8     | 132\*8     | 0.97        | 19.1\*8                 | 1         |

## References

- [FairMOT](https://github.com/ifzhang/FairMOT)

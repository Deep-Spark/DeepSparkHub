# PointRCNN

## Model Description

PointRCNN is a two-stage 3D object detection framework that directly processes raw point cloud data. In the first stage,
it generates accurate 3D box proposals in a bottom-up manner. The second stage refines these proposals using a bin-based
3D box regression loss in canonical coordinates. As the first two-stage detector using only raw point clouds, PointRCNN
achieves state-of-the-art performance on the KITTI dataset, demonstrating superior accuracy in 3D object detection
tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

Download the kitti dataset from <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>

Download the "planes" subdataset from <https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing>

```bash
PointRCNN
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & planes
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```

Generate gt database

```bash
pushd tools/
python3 generate_gt_database.py --class_name 'Car' --split train
popd
```

### Install Dependencies

```bash
## install libGL
yum install -y mesa-libGL

pip3 install easydict tensorboardX shapely fire scikit-image

bash build_and_install.sh

## install numba
pushd numba/
bash clean_numba.sh
bash build_numba.sh
bash install_numba.sh
popd
```

## Model Training

```bash
cd tools/

# Training of RPN stage
## Single GPU training
export CUDA_VISIBLE_DEVICES=0
python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200

## Multiple GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 32 --train_mode rpn --epochs 200 --mgpus

# Training of RCNN stage
## Single GPU training
export CUDA_VISIBLE_DEVICES=0
python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 32 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth

## Multiple GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 32 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth --mgpus

# Evaluation
python3 eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rpn 
python3 eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_70.pth --batch_size 4 --eval_mode rcnn
```

## Model Results

| Model     | GPU        | Stage | FPS         | ACC                   |
|-----------|------------|-------|-------------|-----------------------|
| PointRCNN | BI-V100 x8 | RPN   | 127.56 s/it | iou avg: 0.5417       |
| PointRCNN | BI-V100 x8 | RCNN  | 975.71 s/it | avg detections: 7.243 |

## References

- [PointRCNN](https://github.com/sshaoshuai/PointRCNN)

# PointRCNN

## Model description
PointRCNN 3D object detector to directly generated accurate 3D box proposals from raw point cloud in a bottom-up manner, which are then refined in the canonical coordinate by the proposed bin-based 3D box regression loss. To the best of our knowledge, PointRCNN is the first two-stage 3D object detector for 3D object detection by using only the raw point cloud as input. PointRCNN is evaluated on the KITTI dataset and achieves state-of-the-art performance on the KITTI 3D object detection leaderboard among all published works at the time of submission.

## Installing packages
```bash
## install libGL
yum install mesa-libGL

pip3 install easydict
pip3 install tensorboardX
pip3 install shapely
pip3 install fire
pip3 install scikit-image

bash build_and_install.sh

cd numba
bash clean_numba.sh
bash build_numba.sh
bash install_numba.sh
cd ..
 
```

## Prepare Dataset
Download the kitti dataset from <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>

Download the "planes" subdataset from <https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing>

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
```bash
cd tools
python3 generate_gt_database.py --class_name 'Car' --split train
cd ..
```


## Training
### Training of RPN stage

```bash
# Single GPU training
cd tools
export CUDA_VISIBLE_DEVICES=0
python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --train_mode rpn --epochs 200

# Multiple GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 32 --train_mode rpn --epochs 200 --mgpus
```

### Training of RCNN stage
```bash
# Single GPU training
export CUDA_VISIBLE_DEVICES=0
python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 32 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth

# Multiple GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 32 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth --mgpus
```
## Evaluating
```bash
cd tools
python3 eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rpn 
python3 eval_rcnn.py --cfg_file cfgs/default.yaml --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_200.pth --batch_size 4 --eval_mode rcnn

```

## Training Results


## Reference
https://github.com/sshaoshuai/PointRCNN
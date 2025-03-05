# PointPillars

## Model description
A Simple PointPillars PyTorch Implenmentation for 3D Lidar(KITTI) Detection.

- It can be run without installing [mmcv](https://github.com/open-mmlab/mmcv), [Spconv](https://github.com/traveller59/spconv), [mmdet](https://github.com/open-mmlab/mmdetection) or [mmdet3d](https://github.com/open-mmlab/mmdetection3d). 
- Only one detection network (PointPillars) was implemented in this repo, so the code may be more easy to read. 
- Sincere thanks for the great open-souce architectures [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d), which helps me to learn 3D detetion and implement this repo.

## [Compile] 
```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/zhulf0804/PointPillars.git
cd PointPillars/
git checkout 620e6b0d07e4cb37b7b0114f26b934e8be92a0ba
python3 setup.py build_ext --inplace
pip install .
```

## [Datasets]

1. Download

    Download [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)(29GB), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)(12 GB), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)(16 MB)和[labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)(5 MB)。
    Format the datasets as follows:
    ```
    kitti
        |- ImageSets
            |- train.txt
            |- val.txt
            |- test.txt
            |- trainval.txt
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7418 .bin)
    ```
    The train.txt、val.txt、test.txt and trainval.txt you can get from:
    ```
    wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt
    wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt
    wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt
    wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt
    ```
2. Pre-process KITTI datasets First

    ```
    ln -s path/to/kitti/ImageSets ./dataset
    python3 pre_process_kitti.py --data_root your_path_to_kitti
    ```

    Now, we have datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
            |- velodyne_reduced (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
            |- velodyne_reduced (#7518 .bin)
        |- kitti_gt_database (# 19700 .bin)
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    
    ```

## [Training]

### Single GPU training
```bash
python3 train.py --data_root your_path_to_kitti
```

## Reference
[PointPillars](https://github.com/zhulf0804/PointPillars/tree/620e6b0d07e4cb37b7b0114f26b934e8be92a0ba)
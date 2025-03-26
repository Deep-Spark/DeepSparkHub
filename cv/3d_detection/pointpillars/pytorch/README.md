# PointPillars

## Model Description

PointPillars is an efficient 3D object detection framework designed for LiDAR point cloud data. It organizes point
clouds into vertical columns (pillars) to create a pseudo-image representation, enabling the use of 2D convolutional
networks for processing. This approach balances accuracy and speed, making it suitable for real-time applications like
autonomous driving. PointPillars achieves state-of-the-art performance on the KITTI dataset while maintaining
computational efficiency through its pillar-based encoding and simplified network architecture.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Download:

- [point cloud (29GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
- [images (12 GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
- [calibration files (16 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- [labels (5 MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)

Format the datasets as follows:

```bash
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

```bash
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/test.txt
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/train.txt
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/val.txt
wget https://raw.githubusercontent.com/traveller59/second.pytorch/master/second/data/ImageSets/trainval.txt
```

Pre-process KITTI datasets First.

```bash
ln -s path/to/kitti/ImageSets ./dataset
python3 pre_process_kitti.py --data_root your_path_to_kitti
```

Now, we have datasets as follows:

```bash
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

### Install Dependencies

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

## Model Training

```bash
# Single GPU training
python3 train.py --data_root your_path_to_kitti
```

## References

- [PointPillars](https://github.com/zhulf0804/PointPillars/tree/620e6b0d07e4cb37b7b0114f26b934e8be92a0ba)

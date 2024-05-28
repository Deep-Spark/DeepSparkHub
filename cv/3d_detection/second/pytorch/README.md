# SECOND

## Model description

LiDAR-based or RGB-D-based object detection is used in numerous applications, ranging from autonomous driving to robot vision. Voxel-based 3D convolutional networks have been used for some time to enhance the retention of information when processing point cloud LiDAR data. However, problems remain, including a slow inference speed and low orientation estimation performance. We therefore investigate an improved sparse convolution method for such networks, which significantly increases the speed of both training and inference. We also introduce a new form of angle loss regression to improve the orientation estimation performance and a new data augmentation approach that can enhance the convergence speed and performance. The proposed network produces state-of-the-art results on the KITTI 3D object detection benchmarks while maintaining a fast inference speed.

## Step 1: Installation

```bash
## install libGL and libboost
yum install mesa-libGL
yum install boost-devel

# Install numba
pushd <deepsparkhub_root>/toolbox/numba
python3 setup.py bdist_wheel -d build_pip 2>&1 | tee compile.log
pip3 install build_pip/numba-0.56.4-cp310-cp310-linux_x86_64.whl
popd

# Install spconv
pushd <deepsparkhub_root>/toolbox/spconv
bash clean_spconv.sh
bash build_spconv.sh
bash install_spconv.sh
popd

# Install openpcdet
pushd <deepsparkhub_root>/toolbox/openpcdet
pip3 install -r requirements.txt
bash build_openpcdet.sh
bash install_openpcdet.sh
popd
```

## Step 2: Preparing datasets

Download the kitti dataset from <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>

Download the "planes" subdataset from <https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing>

```bash
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

```bash
# Modify the `DATA_PATH` in the kitti_dataset.yaml to your own
cd <deepsparkhub_root>/toolbox/openpcdet
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## Step 3: Training

### Single GPU training

```bash
cd tools
python3 train.py --cfg_file cfgs/kitti_models/second.yaml
```

### Multiple GPU training

```bash
bash scripts/dist_train.sh 16 --cfg_file cfgs/kitti_models/second.yaml
```

# SECOND-IoU

## Model Description

SECOND-IoU is an enhanced version of the SECOND framework that incorporates Intersection over Union (IoU) optimization
for 3D object detection from LiDAR point clouds. It leverages sparse convolutional networks to efficiently process 3D
data while maintaining spatial information. The model introduces IoU-aware regression to improve bounding box accuracy
and orientation estimation. SECOND-IoU achieves state-of-the-art performance on 3D detection benchmarks, offering faster
inference speeds and better precision than traditional methods, making it suitable for real-time applications like
autonomous driving.

## Model Preparation

### Prepare Resources

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

### Install Dependencies

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

## Model Training

```bash
# Single GPU training
cd tools/
python3 train.py --cfg_file cfgs/kitti_models/second_iou.yaml

# Multiple GPU training
bash scripts/dist_train.sh 16 --cfg_file cfgs/kitti_models/second_iou.yaml
```

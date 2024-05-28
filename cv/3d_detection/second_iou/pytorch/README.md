# SECOND-IoU

## Model description

we present a novel approach called SECOND (Sparsely Embedded CONvolutional Detection), which addresses these challenges in 3D convolution-based detection by maximizing the use of the rich 3D information present in point cloud data. This method incorporates several improvements to the existing convolutional network architecture. Spatially sparse convolutional networks are introduced for LiDAR-based detection and are used to extract information from the z-axis before the 3D data are downsampled to something akin to 2D image data.

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
python3 train.py --cfg_file cfgs/kitti_models/second_iou.yaml
```

### Multiple GPU training

```bash
bash scripts/dist_train.sh 16 --cfg_file cfgs/kitti_models/second_iou.yaml
```

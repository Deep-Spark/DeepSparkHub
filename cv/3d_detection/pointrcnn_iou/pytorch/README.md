# PointRCNN-IoU

## Model Description

PointRCNN-IoU is an enhanced version of the PointRCNN framework that incorporates Intersection over Union (IoU)
optimization for 3D object detection. It processes raw point cloud data in two stages: first generating 3D proposals,
then refining them with IoU-aware regression. This approach improves bounding box accuracy by directly optimizing the
overlap between predicted and ground truth boxes. PointRCNN-IoU maintains the efficiency of its predecessor while
achieving higher precision in 3D object detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.1.1     |  24.06  |

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
python3 train.py --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml

# Multiple GPU training
bash scripts/dist_train.sh 16 --cfg_file cfgs/kitti_models/pointrcnn_iou.yaml
```

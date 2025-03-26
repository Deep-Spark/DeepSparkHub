# Part-A2-Free

## Model Description

Part-A2-Free is an advanced 3D object detection framework for LiDAR point clouds, leveraging part-aware and aggregation
techniques. It operates in two stages: first predicting 3D proposals and intra-object part locations using free part
supervisions, then aggregating these parts to refine box scores and locations. This approach effectively captures object
geometry through a novel RoI-aware point cloud pooling module, achieving state-of-the-art performance on the KITTI
dataset while maintaining computational efficiency for practical applications.

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
cd toolbox/openpcdet
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

### Install Dependencies

```bash
## install libGL and libboost
yum install mesa-libGL
yum install boost-devel

# Install spconv
cd toolbox/spconv
bash clean_spconv.sh
bash build_spconv.sh
bash install_spconv.sh

# Install openpcdet
cd toolbox/openpcdet
pip3 install -r requirements.txt
bash build_openpcdet.sh
bash install_openpcdet.sh
```

## Model Training

```bash
# Single GPU training
cd tools/
python3 train.py --cfg_file cfgs/kitti_models/PartA2_free.yaml

# Multiple GPU training
bash scripts/dist_train.sh 16 --cfg_file cfgs/kitti_models/PartA2_free.yaml
```

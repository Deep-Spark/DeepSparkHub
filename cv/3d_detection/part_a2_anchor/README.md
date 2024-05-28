# Part-A2-Anchor

## Model description
3D object detection from LiDAR point cloud is a challenging problem in 3D scene understanding and has many practical applications. In this paper, we extend our preliminary work PointRCNN to a novel and strong point-cloud-based 3D object detection framework, the part-aware and aggregation neural network (Part-A2 net). The whole framework consists of the part-aware stage and the part-aggregation stage. Firstly, the part-aware stage for the first time fully utilizes free-of-charge part supervisions derived from 3D ground-truth boxes to simultaneously predict high quality 3D proposals and accurate intra-object part locations. The predicted intra-object part locations within the same proposal are grouped by our new-designed RoI-aware point cloud pooling module, which results in an effective representation to encode the geometry-specific features of each 3D proposal. Then the part-aggregation stage learns to re-score the box and refine the box location by exploring the spatial relationship of the pooled intra-object part locations. Extensive experiments are conducted to demonstrate the performance improvements from each component of our proposed framework. Our Part-A2 net outperforms all existing 3D detection methods and achieves new state-of-the-art on KITTI 3D object detection dataset by utilizing only the LiDAR point cloud data.

## Step 1: Installation
```
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

## Step 2: Preparing datasets
Download the kitti dataset from <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>

Download the "planes" subdataset from <https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing>
```
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

```
# Modify the `DATA_PATH` in the kitti_dataset.yaml to your own
cd toolbox/openpcdet
python3 -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

## Step 3: Training

### Single GPU training

```bash
cd tools
python3 train.py --cfg_file cfgs/kitti_models/PartA2.yaml
```

### Multiple GPU training

```bash
bash scripts/dist_train.sh 16 --cfg_file cfgs/kitti_models/PartA2.yaml
```
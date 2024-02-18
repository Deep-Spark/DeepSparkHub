# CenterPoint

## Model description
Three-dimensional objects are commonly represented as 3D boxes in a point-cloud. This representation mimics the well-studied image-based 2D bounding-box detection but comes with additional challenges. Objects in a 3D world do not follow any particular orientation, and box-based detectors have difficulties enumerating all orientations or fitting an axis-aligned bounding box to rotated objects. In this paper, we instead propose to represent, detect, and track 3D objects as points. Our framework, CenterPoint, first detects centers of objects using a keypoint detector and regresses to other attributes, including 3D size, 3D orientation, and velocity. In a second stage, it refines these estimates using additional point features on the object. In CenterPoint, 3D object tracking simplifies to greedy closest-point matching. The resulting detection and tracking algorithm is simple, efficient, and effective. CenterPoint achieved state-of-the-art performance on the nuScenes benchmark for both 3D detection and tracking, with 65.5 NDS and 63.8 AMOTA for a single model. On the Waymo Open Dataset, CenterPoint outperforms all previous single model method by a large margin and ranks first among all Lidar-only submissions.

## Step 1: Installation
```
## install libGL and libboost
yum install mesa-libGL
yum install boost-devel

# Install numba
cd numba
bash clean_numba.sh
bash build_numba.sh
bash install_numba.sh
cd ..

# Install spconv which need cudnn.h
cd spconv
bash clean_spconv.sh
bash build_spconv.sh
bash install_spconv.sh
cd ..

pip3 install -r requirements.txt

bash setup.sh

export PYTHONPATH="${PYTHONPATH}:PATH_TO_CENTERPOINT"
```

## Step 2: Preparing datasets
Download nuScenes from https://www.nuscenes.org/download
```
mkdir -p data/nuscenes
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata

python3 tools/create_data.py nuscenes_data_prep --root-path ./data/nuscenes --version="v1.0-trainval" --nsweeps=10

```


## Step 3: Training

### Single GPU training

```bash
python3 ./tools/train.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py
```

### Multiple GPU training

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py
```

### Evaluation

```bash
python3 ./tools/dist_test.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py --work_dir work_dirs/nusc_centerpoint_voxelnet_01voxel --checkpoint work_dirs/nusc_centerpoint_voxelnet_01voxel/latest.pth
```

## Results

GPUs | FPS | ACC
---- | --- | ---
BI-V100 x8 | 2.423 s/step | mAP: 0.5654


## Reference
- [CenterPoint](https://github.com/tianweiy/CenterPoint)

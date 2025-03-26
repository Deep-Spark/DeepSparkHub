# CenterPoint

## Model Description

CenterPoint is a state-of-the-art 3D object detection and tracking framework that represents objects as points rather
than bounding boxes. It first detects object centers using a keypoint detector, then regresses other attributes like
size, orientation, and velocity. A second stage refines these estimates using additional point features. This approach
simplifies 3D tracking to greedy closest-point matching, achieving top performance on nuScenes and Waymo datasets while
maintaining efficiency and simplicity in implementation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

Download nuScenes from <https://www.nuscenes.org/download>.

```bash
mkdir -p data/nuscenes
# For nuScenes Dataset         
└── NUSCENES_DATASET_ROOT
       ├── samples       <-- key frames
       ├── sweeps        <-- frames without annotation
       ├── maps          <-- unused
       ├── v1.0-trainval <-- metadata

python3 tools/create_data.py nuscenes_data_prep --root-path ./data/nuscenes --version="v1.0-trainval" --nsweeps=10

```

### Install Dependencies

```bash
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

## Model Training

```bash
# Single GPU training
python3 ./tools/train.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py

# Multiple GPU training
python3 -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py

# Evaluation
python3 ./tools/dist_test.py ./configs/nusc/voxelnet/nusc_centerpoint_voxelnet_01voxel.py --work_dir work_dirs/nusc_centerpoint_voxelnet_01voxel --checkpoint work_dirs/nusc_centerpoint_voxelnet_01voxel/latest.pth
```

## Model Results

| Model       | GPU        | FPS          | ACC         |
|-------------|------------|--------------|-------------|
| CenterPoint | BI-V100 x8 | 2.423 s/step | mAP: 0.5654 |

## References

- [CenterPoint](https://github.com/tianweiy/CenterPoint)

# PointNet++

## Model Description

PointNet++ is a hierarchical neural network for processing 3D point cloud data, extending the capabilities of PointNet.
It recursively applies PointNet on nested partitions of the input point set, enabling the learning of local features at
multiple scales. The network adapts to varying point densities through novel set learning layers, improving performance
on complex scenes. PointNet++ excels in tasks like 3D object classification and segmentation by effectively capturing
fine-grained geometric patterns in point clouds.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.06  |

## Model Preparation

### Prepare Resources

```bash
cd data/s3dis/
```

Enter the data/s3dis/ folder, then prepare the dataset according to readme instructions in data/s3dis/ folder.

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

#install mmdetection3d
git clone https://github.com/open-mmlab/mmdetection3d.git -b v1.4.0 --depth=1
cd mmdetection3d
pip install -v -e .
```

## Model Training

```bash
# Single GPU training
python3 tools/train.py configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py

# Multiple GPU training
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/pointnet2/pointnet2_msg_2xb16-cosine-80e_s3dis-seg.py 8
```

## Model Results

| Model      | ceiling | floor  | wall   | beam   | column | window | door   | table  | chair  | sofa   | bookcase | board  | clutter | miou   | acc    | acc_cls |
|------------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|--------|---------|--------|--------|---------|
| PointNet++ | 0.9147  | 0.9742 | 0.7800 | 0.0000 | 0.1881 | 0.5361 | 0.2265 | 0.6922 | 0.8249 | 0.3303 | 0.6585   | 0.5422 | 0.4607  | 0.5483 | 0.8490 | 0.6168  |

## References

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0)

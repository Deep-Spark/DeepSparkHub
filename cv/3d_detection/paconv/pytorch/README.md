# PAConv

## Model Description

PAConv (Position Adaptive Convolution) is an innovative convolution operation for 3D point cloud processing that
dynamically assembles convolution kernels. It constructs kernels by adaptively combining weight matrices from a Weight
Bank, with coefficients learned from point positions through ScoreNet. This data-driven approach provides flexibility to
handle irregular point cloud data efficiently. PAConv integrates seamlessly with existing MLP-based pipelines, achieving
state-of-the-art performance in classification and segmentation tasks while maintaining computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.1     |  24.03  |

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
python3 tools/train.py configs/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py

# Multiple GPU training
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py 8
```

## Model Results

| Model  | ceiling | floor  | wall   | beam   | column | window | door   | table  | chair  | sofa   | bookcase | board  | clutter | miou   | acc    | acc_cls | fps              |
|--------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|--------|---------|--------|--------|---------|------------------|
| PAConv | 0.9488  | 0.9838 | 0.8184 | 0.0000 | 0.1682 | 0.5836 | 0.7387 | 0.7782 | 0.8832 | 0.6101 | 0.7081   | 0.6876 | 0.5810  | 0.6530 | 0.8910 | 0.7131  | 65.3 samples/sec |

## References

- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0)

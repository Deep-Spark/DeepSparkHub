# PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds

## Model description
We introduce Position Adaptive Convolution (PAConv), a generic convolution operation for 3D point cloud processing. The key of PAConv is to construct the convolution kernel by dynamically assembling basic weight matrices stored in Weight Bank, where the coefficients of these weight matrices are self-adaptively learned from point positions through ScoreNet. In this way, the kernel is built in a data-driven manner, endowing PAConv with more flexibility than 2D convolutions to better handle the irregular and unordered point cloud data. Besides, the complexity of the learning process is reduced by combining weight matrices instead of brutally predicting kernels from point positions. Furthermore, different from the existing point convolution operators whose network architectures are often heavily engineered, we integrate our PAConv into classical MLP-based point cloud pipelines without changing network configurations. Even built on simple networks, our method still approaches or even surpasses the state-of-the-art models, and significantly improves baseline performance on both classification and segmentation tasks, yet with decent efficiency. Thorough ablation studies and visualizations are provided to understand PAConv.

## Step 1: Installation

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

## Step 2: Preparing datasets

```bash
cd data/s3dis/
```
Enter the data/s3dis/ folder, then prepare the dataset according to readme instructions in data/s3dis/ folder.

## Step 3: Training

```bash
# Single GPU training
python3 train.py configs/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py

# Multiple GPU training
bash dist_train.sh configs/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py 8
```

## Results

classes | ceiling | floor  | wall   | beam   | column | window | door   | table  | chair  | sofa   | bookcase | board  | clutter | miou   | acc    | acc_cls
---------|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|----------|--------|---------|--------|--------|---------
results | 0.9488  | 0.9838 | 0.8184 | 0.0000 | 0.1682 | 0.5836 | 0.7387 | 0.7782 | 0.8832 | 0.6101 | 0.7081   | 0.6876 | 0.5810  | 0.6530 | 0.8910 | 0.7131

fps = batchsize*8/1batchtime = 65.3 samples/sec

## Reference
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d/tree/v1.4.0)

# PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds

> [PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds](https://arxiv.org/abs/2103.14635)

<!-- [ALGORITHM] -->

## Model description
We introduce Position Adaptive Convolution (PAConv), a generic convolution operation for 3D point cloud processing. The key of PAConv is to construct the convolution kernel by dynamically assembling basic weight matrices stored in Weight Bank, where the coefficients of these weight matrices are self-adaptively learned from point positions through ScoreNet. In this way, the kernel is built in a data-driven manner, endowing PAConv with more flexibility than 2D convolutions to better handle the irregular and unordered point cloud data. Besides, the complexity of the learning process is reduced by combining weight matrices instead of brutally predicting kernels from point positions. Furthermore, different from the existing point convolution operators whose network architectures are often heavily engineered, we integrate our PAConv into classical MLP-based point cloud pipelines without changing network configurations. Even built on simple networks, our method still approaches or even surpasses the state-of-the-art models, and significantly improves baseline performance on both classification and segmentation tasks, yet with decent efficiency. Thorough ablation studies and visualizations are provided to understand PAConv.

## Installing packages
```
## install libGL
yum install mesa-libGL

## install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/
```
```
pip3 install -r requirements/runtime.txt
```
```
#install mmcv v1.7.1
cd deepsparkhub/toolbox/MMDetection/patch/mmcv/v1.7.1
bash build_mmcv.sh
bash install_mmcv.sh
```

## Prepare S3DIS Data
```
cd data/s3dis/
```
Enter the data/s3dis/ folder, then prepare the dataset according to readme instructions in data/s3dis/ folder.

## Training
Single GPU training
```
python3 train.py configs/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py
```
Multiple GPU training
```
bash dist_train.sh configs/paconv/paconv_cuda_ssg_8x8_cosine_200e_s3dis_seg-3d-13class.py 8
```

## Training Results
| Classes | ceiling | floor  | wall   | beam   | column | window | door   | table  | chair  | sofa   | bookcase | board  | clutter | miou   | acc    | acc_cls |
| --------| ------- | -----  | ------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |
| Results |   |  |  |  |  |  |  |  |  |  |    |  |   |   |  |   |

## Reference
https://github.com/open-mmlab/mmdetection3d
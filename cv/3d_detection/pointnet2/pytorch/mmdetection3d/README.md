# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
> [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://arxiv.org/abs/1706.02413)

## Model description
Few prior works study deep learning on point sets. PointNet by Qi et al. is a pioneer in this direction. However, by design PointNet does not capture local structures induced by the metric space points live in, limiting its ability to recognize fine-grained patterns and generalizability to complex scenes. In this work, we introduce a hierarchical neural network that applies PointNet recursively on a nested partitioning of the input point set. By exploiting metric space distances, our network is able to learn local features with increasing contextual scales. With further observation that point sets are usually sampled with varying densities, which results in greatly decreased performance for networks trained on uniform densities, we propose novel set learning layers to adaptively combine features from multiple scales.

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
MMCV_WITH_OPS=1 python3 setup.py develop
```

## Prepare S3DIS Data
```
cd data/s3dis/
```
Enter the data/s3dis/ folder, then prepare the dataset according to readme instructions in data/s3dis/ folder.

## Training
Single GPU training
```
python3 train.py configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py
```
Multiple GPU training
```
bash dist_train.sh configs/pointnet2/pointnet2_msg_16x2_cosine_80e_s3dis_seg-3d-13class.py 8
```

## Training Results
| Classes | ceiling | floor  | wall   | beam   | column | window | door   | table  | chair  | sofa   | bookcase | board  | clutter | miou   | acc    | acc_cls |
| --------| ------- | -----  | ------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |------ |
| Results |  0.9147 | 0.9742 | 0.7800 | 0.0000 | 0.1881 | 0.5361 | 0.2265 | 0.6922 | 0.8249 | 0.3303 | 0.6585   | 0.5422 | 0.4607  |  0.5483 | 0.8490 | 0.6168  |

## Reference
https://github.com/open-mmlab/mmdetection3d
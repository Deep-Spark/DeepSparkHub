# 3D-UNet

## Model Description

3D-UNet is a deep learning model designed for volumetric image segmentation, extending the traditional 2D U-Net
architecture to three dimensions. It effectively processes 3D medical imaging data like CT and MRI scans, learning from
sparsely annotated volumes to produce dense 3D segmentations. The model supports both semi-automated and fully-automated
segmentation workflows, incorporating on-the-fly elastic deformations for efficient data augmentation. 3D-UNet is
particularly valuable in medical imaging for tasks requiring precise 3D anatomical structure delineation.

## Model Preparation

### Prepare Resources

if there is local 'kits19' dataset:

```shell
ln -s /path/to/kits19/ data
```

else:

```shell
bash prepare.sh
```

### Install Dependencies

```shell
pip3 install 'scipy' 'tqdm'
```

## Model Training

```shell
# Single GPU
bash train.sh

# Multi GPU
export NUM_GPUS=8
bash train_dist.sh
```

## Model Results

| GPU        | FP16  | FPS   | Mean dice |
|------------|-------|-------|-----------|
| BI-V100 x8 | False | 11.52 | 0.908     |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 0.908                | SDK V2.2, bs:4, 8x, fp32                 | 12          | 0.908    | 152\*8     | 0.85        | 19.6\*8                 | 1         |

## References

- [mlcommons](https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch)

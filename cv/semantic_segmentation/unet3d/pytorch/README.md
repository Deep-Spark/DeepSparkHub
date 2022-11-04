# 3D-UNet

## Model description

A network for volumetric segmentation that learns from sparsely annotated volumetric images.
Two attractive use cases of this method:
(1) In a semi-automated setup, the user annotates some slices in the volume to be segmented. The network learns from these sparse annotations and provides a dense 3D segmentation.
(2) In a fully-automated setup, we assume that a representative, sparsely annotated training set exists. Trained on this data set, the network densely segments new volumetric images.
The proposed network extends the previous u-net architecture from Ronneberger et al. by replacing all 2D operations with their 3D counterparts. 
The implementation performs on-the-fly elastic deformations for efficient data augmentation during training.
## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'tqdm'

```

### Prepare datasets

if there is local 'kits19' dataset:

```shell

ln -s /path/to/kits19/ data

```
else:

```shell

bash prepare.sh

```


## Step 2: Training

### Single GPU

```shell

bash train.sh

```

### Multi GPU

```shell

bash train_dist.sh <num_gpus>


```

## Results on BI-V100

| GPUs | FP16  | FPS  | Mean dice |
|------|-------| ---- |-----------|
| 1x8  | False | 11.52| 0.908     |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| 0.908                | SDK V2.2,bs:4,8x,fp32                    | 12          | 0.908    | 152\*8     | 0.85        | 19.6\*8                 | 1         |


## Reference

Reference: https://github.com/mlcommons/training/tree/master/image_segmentation/pytorch

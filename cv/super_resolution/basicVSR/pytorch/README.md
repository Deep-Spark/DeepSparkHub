# basicVSR (CVPR2022, Oral)

## Model description

BasicVSR is a video super-resolution pipeline including optical flow and residual blocks. It adopts a typical bidirectional recurrent network. The upsampling module U contains multiple pixel-shuffle and convolutions. In the Figure, red and blue colors represent the backward and forward propagations, respectively. The propagation branches contain only generic components. S, W and R refer to the flow estimation module, spatial warping module, and residual blocks, respectively.

## Step 1: Installing packages

```shell
sh build_env.sh
```

## Step 2: Preparing datasets

Download REDS dataset from [homepage](https://seungjunnah.github.io/Datasets/reds.html)
```shell
mkdir -p data/
ln -s ${REDS_DATASET_PATH} data/REDS
```

## Step 3: Training

### One single GPU
```shell
python3 train.py <config file> [training args]   # config file can be found in the configs directory
```

### Mutiple GPUs on one machine
```shell
bash dist_train.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```
### Example

```shell
bash dist_train.sh configs/basicvsr/basicvsr_reds4.py 8
```
## Reference
https://github.com/open-mmlab/mmediting

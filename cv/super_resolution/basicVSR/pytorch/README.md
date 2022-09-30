# basicVSR (CVPR2022, Oral)

## Model description

BasicVSR is a video super-resolution pipeline including optical flow and residual blocks. It adopts a typical bidirectional recurrent network. The upsampling module U contains multiple pixel-shuffle and convolutions. In the Figure, red and blue colors represent the backward and forward propagations, respectively. The propagation branches contain only generic components. S, W and R refer to the flow estimation module, spatial warping module, and residual blocks, respectively.

## Step 1: Installing packages

```shell
$ sh build_env.sh
```

## Step 2: Preparing datasets


```shell
$ cd /path/to/modelzoo/official/cv/super_resolution/basicVSR/pytorch

# Download REDS to data/REDS  
# Homepage of REDS: https://seungjunnah.github.io/Datasets/reds.html

```

## Step 3: Training

### One single GPU
```shell
$ python3 train.py <config file> [training args]   # config file can be found in the configs directory
```

### Mutiple GPUs on one machine
```shell
$ bash train_dist.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

## Reference
https://github.com/open-mmlab/mmediting

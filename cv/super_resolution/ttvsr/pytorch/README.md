# TTVSR

## Model description

We proposed an approach named TTVSR to study video super-resolution by leveraging long-range frame dependencies. TTVSR introduces Transformer architectures in video super-resolution tasks and formulates video frames into pre-aligned trajectories of visual tokens to calculate attention along trajectories.


## Step 1: Installing packages

```shell
$ pip3 install -r requirements.txt
```

## Step 2: Preparing datasets


```shell
$ cd /path/to/modelzoo/official/cv/super_resolution/ttvsr/pytorch

# Download REDS
$ mkdir -p data/REDS
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

## Results on BI-V100

| GPUs | FP16  | FPS  | PSNR |
|------|-------| ---- | ---- |
| 1x8  | False | 93.9 | 32.12 |


## Reference
https://github.com/researchmm/TTVSR

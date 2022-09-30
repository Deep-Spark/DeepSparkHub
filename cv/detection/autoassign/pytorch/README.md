# AutoAssign

## Model description

Determining positive/negative samples for object detection is known as label assignment. Here we present an anchor-free detector named AutoAssign. It requires little human knowledge and achieves appearance-aware through a fully differentiable weighting mechanism. During training, to both satisfy the prior distribution of data and adapt to category characteristics, we present Center Weighting to adjust the category-specific prior distributions. To adapt to object appearances, Confidence Weighting is proposed to adjust the specific assign strategy of each instance. The two weighting modules are then combined to generate positive and negative weights to adjust each location's confidence. 


## Step 1: Installing packages

```bash
$ pip3 install -r requirements.txt
$ MMCV_WITH_OPS=1 python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
```

## Step 2: Preparing datasets

```bash
$ cd /path/to/modelzoo/cv/detection/autoassign/pytorch
$ mkdir -p data && cd data
# Download from homepage of coco: https://cocodataset.org/
```

## Step 3: Training

### One single GPU

```bash
$ python3 train.py <config file> [training args]   # config file can be found in the configs directory
```

### Multiple GPUs on one machine
```bash
$ bash train_dist.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

## Reference

https://github.com/Megvii-BaseDetection/AutoAssign

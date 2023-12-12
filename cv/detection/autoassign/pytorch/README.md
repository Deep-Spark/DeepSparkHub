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
$ apt install dos2unix
$ mkdir -p data 
$ ln -s /path/to/coco/ ./data
```

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── train2017.txt
├── val2017.txt
└── ...
```

## Step 3: Training

### One single GPU

```bash
$ python3 train.py <config file> [training args]   # config file can be found in the configs directory
```

### Multiple GPUs on one machine
```bash
$ bash dist_train.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

## Reference

https://github.com/Megvii-BaseDetection/AutoAssign

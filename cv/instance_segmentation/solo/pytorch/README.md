# SOLO

## Model description

We present a new, embarrassingly simple approach to instance segmentation in images. Compared to many other dense prediction tasks, e.g., semantic segmentation, it is the arbitrary number of instances that have made instance segmentation much more challenging. In order to predict a mask for each instance, mainstream approaches either follow the 'detect-thensegment' strategy as used by Mask R-CNN, or predict category masks first then use clustering techniques to group pixels into individual instances. We view the task of instance segmentation from a completely new perspective by introducing the notion of "instance categories", which assigns categories to each pixel within an instance according to the instance's location and size, thus nicely converting instance mask segmentation into a classification-solvable problem. Now instance segmentation is decomposed into two classification tasks. We demonstrate a much simpler and flexible instance segmentation framework with strong performance, achieving on par accuracy with Mask R-CNN and outperforming recent singleshot instance segmenters in accuracy. We hope that this very simple and strong framework can serve as a baseline for many instance-level recognition tasks besides instance segmentation.

## Step 1: Installing packages

```bash
$ pip3 install -r requirements.txt
  yum install mesa-libGL
  pip3 install yapf==0.31.0
$ MMCV_WITH_OPS=1 python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
```

## Step 2: Preparing datasets

```bash
$ mkdir -p data/coco
$ cd data/coco
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
bash train.sh
```

### Multiple GPUs on one machine

```bash
$ bash train_dist.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

## Results on BI-V100

Average Precision (AP) @[ loU=0.50:0.95 | area= all | maxDets=1001 ] = 0.361

## Reference

Reference: https://github.com/WXinlong/SOLO

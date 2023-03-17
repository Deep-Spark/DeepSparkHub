# SOLO: Segmenting Objects by Locations

## Model description

We present a new, embarrassingly simple approach to instance segmentation in images. Compared to many other dense prediction tasks, e.g., semantic segmentation, it is the arbitrary number of instances that have made instance segmentation much more challenging. In order to predict a mask for each instance, mainstream approaches either follow the 'detect-thensegment' strategy as used by Mask R-CNN, or predict category masks first then use clustering techniques to group pixels into individual instances. We view the task of instance segmentation from a completely new perspective by introducing the notion of "instance categories", which assigns categories to each pixel within an instance according to the instance's location and size, thus nicely converting instance mask segmentation into a classification-solvable problem. Now instance segmentation is decomposed into two classification tasks. We demonstrate a much simpler and flexible instance segmentation framework with strong performance, achieving on par accuracy with Mask R-CNN and outperforming recent singleshot instance segmenters in accuracy. We hope that this very simple and strong framework can serve as a baseline for many instance-level recognition tasks besides instance segmentation.

## Step 1: Installing packages

```bash
$ pip3 install -r requirements.txt
$ MMCV_WITH_OPS=1 python3 setup.py build && cp build/lib.linux*/mmcv/_ext.cpython* mmcv
```

## Step 2: Preparing datasets

```bash
$ mkdir -p data/coco
$ cd data/coco
$ wget http://images.cocodataset.org/zips/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip annotations_trainval2017.zip
$ unzip train2017.zip
$ unzip val2017.zip
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

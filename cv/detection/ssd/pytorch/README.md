# SSD

## Model description

We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300x300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500x500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model. Code is available at https://github.com/weiliu89/caffe/tree/ssd .

## Prepare

### Download dataset

```
$ mkdir -p /home/data/perf/ssd
$ cd /home/data/perf/ssd
$ wget http://images.cocodataset.org/zips/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip annotations_trainval2017.zip
$ unzip train2017.zip
$ unzip val2017.zip
```

### Download backbone
```
cd /home/data/perf/ssd
$ wget https://download.pytorch.org/models/resnet34-333f7ec4.pth
```


### Training

### Multiple GPUs on one machine

```
$ cd /path/to/model_zoo/cv/detection/ssd/pytorch/base
$ source ../iluvatar/config/environment_variables.sh
$ python3  prepare.py --name iluvatar --data_dir /home/data/perf/ssd
$ bash run_training.sh --name iluvatar --config V100x1x8 --data_dir /home/data/perf/ssd --backbone_path /home/data/perf/ssd/resnet34-333f7ec4.pth
```

## Results on BI-V100

| GPUs | Batch Size | FPS | Train Epochs | mAP  |
|------|------------|-----|--------------|------|
| 1x8  | 192        | 2858 | 65           | 0.23 |



## Reference
https://github.com/mlcommons/training_results_v0.7/tree/master/NVIDIA/benchmarks/ssd/implementations/pytorch
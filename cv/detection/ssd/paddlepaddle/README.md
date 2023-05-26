# SSD
## Model description
We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300x300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500x500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model. Code is available at https://github.com/weiliu89/caffe/tree/ssd .

## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleDetection.git
```

```
cd PaddleDetection
pip3 install -r requirements.txt
```

## Step 2: Prepare Datasets

```bash
python3 dataset/coco/download_coco.py
```

## Step 3: Training
Notice: modify configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml file, modify the datasets path as yours.
```
cd PaddleDetection
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1
python3 -u -m paddle.distributed.launch --gpus 0,1 tools/train.py -c configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml --eval
```

## Results on BI-V100

<div align="center">

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 2 cards     | bbox=73.62,FPS=45.49,BatchSize=32    |

</div>

## Reference
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

# SSD

## Model description

We present a method for detecting objects in images using a single deep neural network. Our approach, named SSD, discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location. At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape. Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes. Our SSD model is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stage and encapsulates all computation in a single network. This makes SSD easy to train and straightforward to integrate into systems that require a detection component. Experimental results on the PASCAL VOC, MS COCO, and ILSVRC datasets confirm that SSD has comparable accuracy to methods that utilize an additional object proposal step and is much faster, while providing a unified framework for both training and inference. Compared to other single stage methods, SSD has much better accuracy, even with a smaller input image size. For 300x300 input, SSD achieves 72.1% mAP on VOC2007 test at 58 FPS on a Nvidia Titan X and for 500x500 input, SSD achieves 75.1% mAP, outperforming a comparable state of the art Faster R-CNN model. Code is available at https://github.com/weiliu89/caffe/tree/ssd .

## Prepare

### Download the VOC dataset
```
cd dataset
```
Download[ Pascal VOC Dataset](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) and reorganize the directory as follows:
```
VOCROOT/
		   |->VOC2007/
		   |    |->Annotations/
		   |    |->ImageSets/
		   |    |->...
		   |->VOC2012/   # use it
		   |    |->Annotations/
		   |    |->ImageSets/
		   |    |->...
		   |->VOC2007TEST/
		   |    |->Annotations/
		   |    |->...
```
VOCROOT is your path of the Pascal VOC Dataset.
```
mkdir tfrecords
pip3 install tf_slim
python3 convert_voc_sample_tfrecords.py --dataset_directory=./ --output_directory=tfrecords --train_splits VOC2012_sample --validation_splits VOC2012_sample

cd ..
```
### Download the checkpoint
Download the pre-trained VGG-16 model (reduced-fc) from [here](https://drive.google.com/drive/folders/184srhbt8_uvLKeWW_Yo8Mc5wTyc0lJT7) and put them into one sub-directory named 'model' (we support SaverDef.V2 by default, the V1 version is also available for sake of compatibility).

### Train
#### multi gpus
```
python3 train_ssd.py --batch_size 16
````


## Result

|               | acc      |       fps |
| ---           | ---       | ---       |
|    multi_card |  0.783513   | 3.177  |

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
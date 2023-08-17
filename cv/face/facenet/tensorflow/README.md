## Facenet
## Model description
This is a facenet-tensorflow library that can be used to train your own face recognition model.

## Step 1: Installation
```bash
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training. This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance improvement has been seen if the dataset has been filtered before training. Some more information about how this was done will come later.
The best performing model has been trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

## download  

```bash
cd data
download dataset in this way: 
download link: https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw   password: bcrq
The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training
```

## Pre-processing
### Face alignment using MTCNN
One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial occlusion, silhouettes, etc). This makes the training set too "easy" which causes the model to perform worse on other benchmarks.
To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to work very well in this setting is the
[Multi-task CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found [here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very good results. A Python/Tensorflow implementation of MTCNN can be found [here](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not give identical results to the Matlab/Caffe implementation but the performance is very similar.

## Step 3: Training
Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using softmax loss on the CASIA-WebFace dataset can be found on the page [Classifier training of Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1) and .

# One Card
```bash
nohup bash train_facenet.sh 1> train_facenet.log 2> train_facenet_error.log & tail -f train_facenet.log
```

## Results

|   model |    FPS | LFW_Accuracy     |
|---------|--------| -----------------|
| facenet | 216.96 | 0.98900+-0.00642 |

## Reference
https://github.com/davidsandberg/facenet

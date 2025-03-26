# Facenet

## Model Description

Facenet is a deep learning model for face recognition that directly maps face images to a compact Euclidean space, where
distances correspond to face similarity. It uses a triplet loss function to ensure that faces of the same person are
closer together than those of different individuals. Facenet excels in tasks like face verification, recognition, and
clustering, offering high accuracy and efficiency. Its compact embeddings make it scalable for large-scale applications
in security and identity verification.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training.
This training set consists of total of 453 453 images over 10 575 identities after face detection. Some performance
improvement has been seen if the dataset has been filtered before training. Some more information about how this was
done will come later. The best performing model has been trained on the
[VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset consisting of ~3.3M faces and ~9000 classes.

Download from [Baidu YunPan](https://pan.baidu.com/s/1qMxFR8H_ih0xmY-rKgRejw) with password 'bcrq'.

The [CASIA-WebFace](http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html) dataset has been used for training.

```bash
$ ls data/webface_182_44
0000045
...

$ ls data/lfw_data
lfw  lfw_160  lfw.tgz
```

Pre-processing.

Face alignment using MTCNN.

One problem with the above approach seems to be that the Dlib face detector misses some of the hard examples (partial
occlusion, silhouettes, etc). This makes the training set too "easy" which causes the model to perform worse on other
benchmarks. To solve this, other face landmark detectors has been tested. One face landmark detector that has proven to
work very well in this setting is the [Multi-task
CNN](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html). A Matlab/Caffe implementation can be found
[here](https://github.com/kpzhang93/MTCNN_face_detection_alignment) and this has been used for face alignment with very
good results. A Python/Tensorflow implementation of MTCNN can be found
[here](https://github.com/davidsandberg/facenet/tree/master/src/align). This implementation does not give identical
results to the Matlab/Caffe implementation but the performance is very similar.

### Install Dependencies

```bash
# Install requirements.
bash init.sh
pip3 install numpy==1.23.5
```

## Model Training

Currently, the best results are achieved by training the model using softmax loss. Details on how to train a model using
softmax loss on the CASIA-WebFace dataset can be found on the page [Classifier training of
Inception-ResNet-v1](https://github.com/davidsandberg/facenet/wiki/Classifier-training-of-inception-resnet-v1) and .

```bash
# One Card
nohup bash train_facenet.sh 1> train_facenet.log 2> train_facenet_error.log & tail -f train_facenet.log

# Multiple cards (DDP)
## 8 Cards(DDP)
bash train_facenet_ddp.sh
```

## Model Results

| Model   | FPS    | LFW_Accuracy     |
|---------|--------|------------------|
| Facenet | 216.96 | 0.98900+-0.00642 |

## References

- [facenet](https://github.com/davidsandberg/facenet)

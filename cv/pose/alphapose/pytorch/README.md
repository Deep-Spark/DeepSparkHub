
### AlphaPose

AlphaPose is an accurate multi-person pose estimator, which is the first open-source system that achieves 70+ mAP (75 mAP) on COCO dataset and 80+ mAP (82.1 mAP) on MPII dataset. To match poses that correspond to the same person across frames, we also provide an efficient online pose tracker called Pose Flow. It is the first open-source online pose tracker that achieves both 60+ mAP (66.5 mAP) and 50+ MOTA (58.3 MOTA) on PoseTrack Challenge dataset.

## Install requirements

```bash
## install libGL
yum install mesa-libGL

## install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/
```

```
pip3 install seaborn pandas pycocotools matplotlib
pip3 install easydict tensorboardX opencv-python
```

### 1.Dataset: 

#### TrainData(coco)
    $ cd /home/datasets/cv 
## Prepare datasets

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

```bash
mkdir -p /home/datasets/cv/
ln -s /path/to/coco2017 /home/datasets/cv/coco
```


### 1.Train:
    $ bash ./scripts/trainval/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml 1

### result
|       model       |     GPU     | FP32                                 | 
|-------------------| ----------- | ------------------------------------ |
|   AlphaPose       | 8 cards     | acc=84.29                            |


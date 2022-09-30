#!/bin/bash


mkdir -p coco/images
cd coco/images

# Download Images
wget -c http://10.150.9.95/swapp/datasets/cv/detection/coco2014/train2014.zip
wget -c http://10.150.9.95/swapp/datasets/cv/detection/coco2014/val2014.zip
wget -c http://10.150.9.95/swapp/datasets/cv/detection/coco2014/labels.tgz

# Unzip
unzip -q train2014.zip
unzip -q val2014.zip
tar xzf labels.tgz

cd ..
wget -c "https://pjreddie.com/media/files/coco/5k.part"
wget -c "https://pjreddie.com/media/files/coco/trainvalno5k.part"


# Set Up Image Lists
paste <(awk "{print \"$PWD\"}" <5k.part) 5k.part | tr -d '\t' > 5k.txt
paste <(awk "{print \"$PWD\"}" <trainvalno5k.part) trainvalno5k.part | tr -d '\t' > trainvalno5k.txt

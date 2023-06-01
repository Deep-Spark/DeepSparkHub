# MMdetection

## Environment

```
cd mmcv
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
cd ../mmdetection
bash clean_mmdetection.sh
bash build_mmdetection.sh
cd build_pip 
# example please install your whl
pip3 install mmdet-2.22.0+corex20221122024846-py3-none-any.whl
pip3 install yapf
pip3 install addict
pip3 install opencv-python
yum install mesa-libGL
```

## Step 2: Preparing datasets

```
$ mkdir -p <project_path>/datasets/coco
$ cd <project_path>/datasets/coco
$ wget http://images.cocodataset.org/zips/annotations_trainval2017.zip
$ wget http://images.cocodataset.org/zips/train2017.zip
$ wget http://images.cocodataset.org/zips/val2017.zip
$ unzip annotations_trainval2017.zip
$ unzip train2017.zip
$ unzip val2017.zip
```

```
mkdir data
cd data
ln -s <project_path>/datasets/coco coco
cd ..
```



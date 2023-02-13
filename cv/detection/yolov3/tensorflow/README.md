## Prepare
```
bash init_tf.sh
```

## Download dataset and checkpoint
```
# download dataset
mkdir -p VOC
cd VOC
wget http://10.150.9.95/swapp/datasets/cv/detection/VOC_07_12.tgz
tar -zxvf VOC_07_12.tgz
rm -rf VOC_07_12.tgz
cd ..

# download checkpoint
mkdir checkpoint
cd checkpoint
wget http://10.150.9.95/swapp/datasets/cv/detection/yolov3_coco_demo.ckpt.tar.gz
tar -zxvf yolov3_coco_demo.ckpt.tar.gz
rm -rf yolov3_coco_demo.ckpt.tar.gz
cd ..
```

## Run training 
```
bash ./run_training.sh
```

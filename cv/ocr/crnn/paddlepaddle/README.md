# CRNN


## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

```
cd PaddleOCR
pip3 install -r requirements.txt
```

## Step 2: Prepare Datasets
Download [ICDAR2015](https://paperswithcode.com/dataset/icdar-2015), then the icdar-2015 path as follows:
```
root@AE-ubuntu:/home/datasets/ICDAR2015/text_localization# ls -al /home/datasets/ICDAR2015/text_localization
total 133420
drwxr-xr-x 4 root root      179 Jul 21 15:54 .
drwxr-xr-x 3 root root       39 Jul 21 15:50 ..
drwxr-xr-x 2 root root    12288 Jul 21 15:53 ch4_test_images
-rw-r--r-- 1 root root 44359601 Jul 21 15:51 ch4_test_images.zip
-rw-r--r-- 1 root root 90667586 Jul 21 15:51 ch4_training_images.zip
drwxr-xr-x 2 root root    24576 Jul 21 15:53 icdar_c4_train_imgs
-rw-r--r-- 1 root root   468453 Jul 21 15:54 test_icdar2015_label.txt
-rw-r--r-- 1 root root  1063118 Jul 21 15:54 train_icdar2015_label.txt
```

## Step 3: Training
Notice: modify configs/ppyoloe/ppyoloe_crn_s_300e_coco.yml file, modify the datasets path as yours.
```
cd PaddleOCR
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7  tools/train.py -c configs/rec/rec_mv3_none_bilstm_ctc.yml Global.use_visualdl=True
```

## Reference
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
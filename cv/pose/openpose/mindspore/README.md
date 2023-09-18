# Openpose

## Model description

Openpose network proposes a bottom-up human attitude estimation algorithm using Part Affinity Fields (PAFs). Instead of a top-down algorithm: Detect people first and then return key-points and skeleton. The advantage of openpose is that the computing time does not increase significantly as the number of people in the image increases.However,the top-down algorithm is based on the detection result, and the runtimes grow linearly with the number of people.

[Paper](https://arxiv.org/abs/1611.08050): Zhe Cao,Tomas Simon,Shih-En Wei,Yaser Sheikh,"Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields",The IEEE Conference on Computer Vision and Pattern Recongnition(CVPR),2017

## Step 1:Installation

```
# Pip the requirements
pip3 install -r requirements.txt
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7/
./configure --prefix=/usr/local/bin --with-orte
make -j4 && make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

## Step 2:Preparing datasets

* Go to visit [COCO official website](https://gitee.com/link?target=https%3A%2F%2Fcocodataset.org%2F%23download), then select the COCO dataset you want to download. Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

  ```
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
* Create the mask dataset. Run python gen_ignore_mask.py

  ```
  python3 ./src/gen_ignore_mask.py --train_ann ./coco2017/annotations/person_keypoints_train2017.json --val_ann ./coco2017/annotations/person_keypoints_val2017.json --train_dir ./coco2017/train2017 --val_dir ./coco2017/val2017
  ```
* The dataset folder is generated in the root directory and contains the following files:

  ```
  ├── coco2017
      ├── annotations
          ├─ person_keypoints_train2017.json
          └─ person_keypoints_val2017.json
      ├─ ignore_mask_train
      ├─ ignore_mask_val
      ├─ train2017
      ├─ val2017
      └─ ...
  ```
* Download the VGG19 model of the MindSpore version:

  [vgg19-0-97_5004.ckpt](https://download.mindspore.cn/model_zoo/converted_pretrained/vgg/vgg19-0-97_5004.ckpt)

## Step 3:Training

Change the absolute path of the data in running shell `train_openpose_coco2017_1card.sh`  `train_openpose_coco2017_8card.sh`.

For example in `train_openpose_coco2017_1card.sh`:

```
bash scripts/run_standalone_train.sh /home/coco2017/train2017 /home/coco2017/annotations/person_keypoints_train2017.json /home/coco2017/ignore_mask_train /home/vgg19-0-97_5004.ckpt
```

```
# Run on 1 GPU
bash train_openpose_coco2017_1card.sh

# Run on 8 GPU 
bash train_openpose_coco2017_8card.sh

# Run eval
python3 eval.py --model_path /home/openpose_train_8gpu_ckpt/0-80_663.ckpt --imgpath_val coco2017/val2017 --ann coco2017/annotations/person_keypoints_val2017.json
```

## Results

| GPUS       | AP     | AP  .5 | AR     | AR  .5 |
| ---------- | ------ | ------- | ------ | ------- |
| BI V100×8 | 0.3979 | 0.6654  | 0.4435 | 0.6889  |

## Reference

[Openpose](https://gitee.com/mindspore/models/tree/master/official/cv/OpenPose)

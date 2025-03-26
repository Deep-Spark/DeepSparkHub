# OpenPose

## Model Description

OpenPose is a real-time multi-person 2D pose estimation model that uses a bottom-up approach with Part Affinity Fields
(PAFs) to detect human body keypoints and their connections. Unlike top-down methods, OpenPose's computational
efficiency remains stable regardless of the number of people in an image. It simultaneously detects body parts and
associates them to individuals, making it particularly effective for scenarios with multiple people, such as crowd
analysis and human-computer interaction applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

- Go to visit [COCO official website](https://gitee.com/link?target=https%3A%2F%2Fcocodataset.org%2F%23download), then
  select the COCO dataset you want to download. Take coco2017 dataset as an example, specify `/path/to/coco2017` to your
  COCO path in later training process, the unzipped dataset path structure sholud look like:

```bash
  coco2017
  ├── annotations
  │   ├── instances_train2017.json
  │   ├── instances_val2017.json
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

- Create the mask dataset. Run python gen_ignore_mask.py

```bash
python3 ./src/gen_ignore_mask.py --train_ann ./coco2017/annotations/person_keypoints_train2017.json --val_ann ./coco2017/annotations/person_keypoints_val2017.json --train_dir ./coco2017/train2017 --val_dir ./coco2017/val2017
```

- The dataset folder is generated in the root directory and contains the following files:

```bash
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

- Download the VGG19 model of the MindSpore version:

[vgg19-0-97_5004.ckpt](https://download.mindspore.cn/model_zoo/converted_pretrained/vgg/vgg19-0-97_5004.ckpt)

### Install Dependencies

```bash
# Pip the requirements
pip3 install -r requirements.txt
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7/
./configure --prefix=/usr/local/bin --with-orte
make -j4 && make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

## Model Training

Change the absolute path of the data in running shell `train_openpose_coco2017_1card.sh`  `train_openpose_coco2017_8card.sh`.

For example in `train_openpose_coco2017_1card.sh`:

```bash
bash scripts/run_standalone_train.sh /home/coco2017/train2017 /home/coco2017/annotations/person_keypoints_train2017.json /home/coco2017/ignore_mask_train /home/vgg19-0-97_5004.ckpt
```

```bash
# Run on 1 GPU
bash train_openpose_coco2017_1card.sh

# Run on 8 GPU 
bash train_openpose_coco2017_8card.sh

# Run eval
python3 eval.py --model_path /home/openpose_train_8gpu_ckpt/0-80_663.ckpt --imgpath_val coco2017/val2017 --ann coco2017/annotations/person_keypoints_val2017.json
```

## Model Results

| GPUS       | AP     | AP  .5 | AR     | AR  .5 |
|------------|--------|--------|--------|--------|
| BI-V100 ×8 | 0.3979 | 0.6654 | 0.4435 | 0.6889 |

## References

- [Openpose](https://gitee.com/mindspore/models/tree/master/official/cv/OpenPose)

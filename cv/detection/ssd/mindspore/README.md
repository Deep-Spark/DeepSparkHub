# SSD

## Model Description

SSD (Single Shot MultiBox Detector) is a fast and efficient object detection model that predicts bounding boxes and
class scores in a single forward pass. It uses a set of default boxes at different scales and aspect ratios across
multiple feature maps to detect objects of various sizes. SSD combines predictions from different layers to handle
objects at different resolutions, offering a good balance between speed and accuracy for real-time detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Download dataset in /home/datasets/cv/coco2017

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant
domain/network architecture. In the following sections, we will introduce how to run the scripts using the related
dataset below.

Dataset used: [COCO2017](<http://images.cocodataset.org/>)

- Dataset size：19G
  - Train：18G，118000 images  
  - Val：1G，5000 images
  - Annotations：241M，instances，captions，person_keypoints etc
- Data format：image and json files
  - Note：Data will be processed in dataset.py

Change the `coco_root` and other settings you need in `src/config.py`. The directory structure is as follows:

```bash
.
└─coco_dataset
  ├─annotations
    ├─instance_train2017.json
    └─instance_val2017.json
  ├─val2017
  └─train2017
```

If your own dataset is used. **Select dataset to other when run script.**
    Organize the dataset information into a TXT file, each row in the file is as follows:

```bash
train2017/0000001.jpg 0,259,401,459,7 35,28,324,201,2 0,30,59,80,2
```

Each row is an image annotation which split by space, the first column is a relative path of image, the others are box
and class infomations of the format [xmin,ymin,xmax,ymax,class]. We read image from an image path joined by the
`image_dir`(dataset directory) and the relative path in `anno_path`(the TXT file path), `image_dir` and `anno_path` are
setting in `src/config.py`.

Download [resnet50.ckpt](https://pan.baidu.com/s/1rrhsZqDVmNxR-bCnMPvFIw?pwd=8766).

```bash
mv resnet50.ckpt ./ckpt
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
pip3 install easydict
```

## Model Training

```bash
mpirun -allow-run-as-root -n 8 --output-filename log_output --merge-stderr-to-stdout \
python3 train.py --run_distribute=True \
                 --lr=0.05 \
                 --dataset=coco \
                 --device_num=8 \
                 --loss_scale=1 \
                 --device_target="GPU" \
                 --epoch_size=60 \
                 --config_path=./config/ssd_resnet50_fpn_config_gpu.yaml \
                 --output_path './output' > log.txt 2>&1 &
```

## Model Results on BI-V100

| Model | GPU         | per step time | MAP   |
|-------|-------------|---------------|-------|
| SSD   | BI-V100 x8  | 0.814s        | 0.374 |
| SSD   | NV-V100s x8 | 0.797s        | 0.369 |

## References

- [Paper](https://arxiv.org/abs/1512.02325)

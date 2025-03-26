# PP-HumanSegV2

## Model Description

PP-HumanSegV2 is an advanced deep learning model for human segmentation, specializing in both portrait and general human
segmentation tasks. Developed by PaddleSeg, it offers improved accuracy, faster inference speed, and enhanced robustness
compared to its predecessor. The model supports zero-cost deployment for immediate use in products and allows
fine-tuning for better performance. PP-HumanSegV2 is particularly effective for applications like video background
replacement and portrait segmentation, delivering high-quality results with optimized computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Go to visit [PP-HumanSeg14K official website](https://paperswithcode.com/dataset/pp-humanseg14k), then download the
PP-HumanSeg14K dataset, or you can download via [Baidu Netdisk](https://pan.baidu.com/s/1Buy74e5ymu2vXYlYfGvBHg)
password: vui7 , [Google Cloud Disk](https://drive.google.com/file/d/1eEIV9lM2Kl1Ejcj3Cuht8EHN5eNF8Zjn/view?usp=sharing)

The dataset path structure sholud look like:

```bash
PP-HumanSeg14K/
├── annotations
│   ├── train
│   └── val
└── images
│   ├── train
│   └── val
│   └── test
└──train.txt
└──val.txt
└──test.txt
└──LICENSE
└──README.txt

```

### Install Dependencies

```bash
git clone -b develop https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 
pip3 install urllib3==1.26.6
yum install mesa-libGL
python3 setup.py develop
```

## Model Training

Change ./contrib/PP-HumanSeg/configs/portrait_pp_humansegv2_lite.yml dataset path as your dateset path

```bash
# One GPU
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config contrib/PP-HumanSeg/configs/portrait_pp_humansegv2_lite.yml --save_dir output/human_pp_humansegv2_lite --save_interval 500 --do_eval --use_vdl

# Eight GPUs
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py \
       --config contrib/PP-HumanSeg/configs/portrait_pp_humansegv2_lite.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500
```

## Model Results

| MODEL         | GPU        | mIoU  | Acc    | Kappa  | Dice   | FPS     |
|---------------|------------|-------|--------|--------|--------|---------|
| PP-HumanSegV2 | BI-V100 x8 | 0.798 | 0.9860 | 0.9642 | 0.9821 | 34.0294 |

## References

- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

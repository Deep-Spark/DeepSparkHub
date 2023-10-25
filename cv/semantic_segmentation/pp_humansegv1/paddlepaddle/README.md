# PP-HumanSegV1

## Model description

Human segmentation is a high-frequency application in the field of image segmentation.
Generally, human segentation can be classified as portrait segmentation and general human segmentation.

For portrait segmentation and general human segmentation, PaddleSeg releases the PP-HumanSeg models, which has **good performance in accuracy, inference speed and robustness**. Besides, we can deploy PP-HumanSeg models to products without training
Besides, PP-HumanSeg models can be deployed to products at zero cost, and it also support fine-tuning to achieve better performance.

The following is demonstration videos (due to the video is large, the loading will be slightly slow) .We provide full-process application guides from training to deployment, as well as video streaming segmentation and background replacement tutorials. Based on Paddle.js, you can experience the effects of [Portrait Snapshot](https://paddlejs.baidu.com/humanseg), [Video Background Replacement and Barrage Penetration](https://www.paddlepaddle.org.cn/paddlejs).

PP-HumanSegV1-Lite protrait segmentation model: It has good performance in accuracy and model size and the model architecture in [url](https://github.com/PaddlePaddle/PaddleSeg/tree/develop/configs/pp_humanseg_lite).

## Step 1: Installation

```bash
git clone -b develop https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 
pip3 install urllib3==1.26.6
yum install mesa-libGL
python3 setup.py develop
```

## Step 2: Preparing datasets

Go to visit [PP-HumanSeg14K official website](https://paperswithcode.com/dataset/pp-humanseg14k), then download the PP-HumanSeg14K dataset, or you can download via [Baidu Netdisk](https://pan.baidu.com/s/1Buy74e5ymu2vXYlYfGvBHg) password: vui7 , [Google Cloud Disk](https://drive.google.com/file/d/1eEIV9lM2Kl1Ejcj3Cuht8EHN5eNF8Zjn/view?usp=sharing)

The dataset path structure sholud look like:

```
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


## Step 3: Training

```bash
# Change ./contrib/PP-HumanSeg/configs/portrait_pp_humansegv1_lite.yml dataset path as your dateset path 

# One GPU
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --config contrib/PP-HumanSeg/configs/portrait_pp_humansegv1_lite.yml --save_dir output/human_pp_humansegv1_lite --save_interval 500 --do_eval --use_vdl

# Eight GPUs
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py  \
       --config contrib/PP-HumanSeg/configs/portrait_pp_humansegv1_lite.yml \
       --do_eval \
       --use_vdl \
       --save_interval 500
```

## Results

| MODEL         | mIoU    |Acc     | Kappa  |Dice   |
| ----------    | ------  |------  |--------|-----  |
| pp_humansegv1 | 0.9591  |0.9836  |0.9581  |0.9790 |

| GPUS       | FPS     | 
| ---------- | ------  |
| BI-V100x 8 | 24.54   |

## Reference
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
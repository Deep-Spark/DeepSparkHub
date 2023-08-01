# PP-PicoDet

## Model description
    PicoDet is an ultra lightweight real-time object detection model that includes four different sizes of XS/S/M/L. 
    By using structures such as TAL, ETA Head, and PAN, the accuracy of the model is improved. When deploying model inference, 
    the model supports including post-processing in the network, thereby supporting direct output of prediction results.

## Step 1:Installation
```bash
git clone https://github.com/PaddlePaddle/PaddleDetection.git
cd PaddleDetection
pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python3 setup.py install
```

## Step 2:Preparing datasets
    Download the Coco datasets, Specify coco2017 to your Coco path in later training process. 
    The Coco datesets path structure should look like(assuming coco2017 is used):
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

## Step 3:Training

assuming we are going to train picodet-l, the model config file is 'configs/picodet/picodet_l_640_coco_lcnet.yml'
vim configs/datasets/coco_detection.yml, set 'dataset_dir' in the configuration file to coco2017, then start trainging.

single gpu:
```bash
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py -c configs/picodet/picodet_l_640_coco_lcne.yml --eval
```
multi-gpu:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/picodet/picodet_l_640_coco_lcne.yml --eval
```

## Results

| GPUs        | IPS       | mAP0.5:0.95  | mAP0.5       |
|-------------|-----------|--------------|--------------|
| BI-V100 x 8 | 19.84     | 41.2         | 58.2         |

## Reference:

https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/configs/picodet/README_en.md
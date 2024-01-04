# DNLNet

## Model description
[Disentangled Non-Local Neural Networks](https://arxiv.org/abs/2006.06668) Minghao Yin, Zhuliang Yao, Yue Cao, Xiu Li, Zheng Zhang, Stephen Lin, Han Hu


## Step 1: Installing
```
git clone -b release/2.7 https://github.com/PaddlePaddle/PaddleSeg.git
cd PaddleSeg
pip3 install -r requirements.txt
pip3 install protobuf==3.20.3 
pip3 install urllib3==1.26.6
yum install mesa-libGL
```

## Step 2: Prepare Datasets

Go to visit [Cityscapes official website](https://www.cityscapes-dataset.com/), then choose 'Download' to download the Cityscapes dataset.

Specify `/path/to/cityscapes` to your Cityscapes path in later training process, the unzipped dataset path structure sholud look like:

```bash
cityscapes/
├── gtFine
│   ├── test
│   ├── train
│   │   ├── aachen
│   │   └── bochum
│   └── val
│       ├── frankfurt
│       ├── lindau
│       └── munster
└── leftImg8bit
    ├── train
    │   ├── aachen
    │   └── bochum
    └── val
        ├── frankfurt
        ├── lindau
        └── munster
```

Datasets preprocessing:
```bash
pip3 install cityscapesscripts

python3 tools/data/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8

python3 tools/data/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","
```

then the cityscapes path as follows:
```
root@5574247e63f8:~# ls -al /path/to/cityscapes
total 11567948
drwxr-xr-x 4 root root         227 Jul 18 03:32 .
drwxr-xr-x 6 root root         179 Jul 18 06:48 ..
-rw-r--r-- 1 root root         298 Feb 20  2016 README
drwxr-xr-x 5 root root          58 Jul 18 03:30 gtFine
-rw-r--r-- 1 root root   252567705 Jul 18 03:22 gtFine_trainvaltest.zip
drwxr-xr-x 5 root root          58 Jul 18 03:30 leftImg8bit
-rw-r--r-- 1 root root 11592327197 Jul 18 03:27 leftImg8bit_trainvaltest.zip
-rw-r--r-- 1 root root        1646 Feb 17  2016 license.txt
-rw-r--r-- 1 root root      193690 Jul 18 03:32 test.txt
-rw-r--r-- 1 root root      398780 Jul 18 03:32 train.txt
-rw-r--r-- 1 root root       65900 Jul 18 03:32 val.txt
```

## Step 3: Training
Notice: modify configs/dnlnet/dnlnet_resnet50_os8_cityscapes_1024x512_80k.yml file, modify the datasets path as yours.
```
cd PaddleSeg
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=4,5,6,7 # setup 4 available cards
python3 -u -m paddle.distributed.launch train.py \
       --config configs/dnlnet/dnlnet_resnet50_os8_cityscapes_1024x512_80k.yml \
       --do_eval \
       --precision fp16 \
       --amp_level O1
```

## Reference
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)

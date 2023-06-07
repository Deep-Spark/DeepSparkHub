# UNet
## Model description
A network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently.
The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. 

## Step 1: Installing
```
git clone https://github.com/PaddlePaddle/PaddleSeg.git
```

```
cd PaddleSeg
pip3 install -r requirements.txt
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

python3 tools/convert_cityscapes.py --cityscapes_path /path/to/cityscapes --num_workers 8

python3 tools/create_dataset_list.py /path/to/cityscapes --type cityscapes --separator ","
```

then the cityscapes path as follows:

```bash
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
Notice: modify configs/ssd/ssd_mobilenet_v1_300_120e_voc.yml file, modify the datasets path as yours.
The training is use AMP model.
```
cd PaddleSeg
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -u -m paddle.distributed.launch --gpus 0,1,2,3 train.py \
       --config configs/unet/unet_cityscapes_1024x512_160k.yml \
       --do_eval \
       --use_vdl \
       --save_dir output_unet_amp \
       --precision fp16 \
       --amp_level O1
```

## Reference
- [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
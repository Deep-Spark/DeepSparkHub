# YOLACT

## Model Description

YOLACT (You Only Look At Coefficients) is a real-time instance segmentation model that separates mask prediction from
object detection. It generates prototype masks and prediction coefficients independently, then combines them to produce
instance masks. This approach enables fast processing while maintaining competitive accuracy. YOLACT++ further enhances
performance with deformable convolutions and optimized prediction heads. The model achieves real-time speeds on single
GPUs, making it suitable for applications requiring quick instance segmentation in video streams or interactive systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

```bash
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

### Install Dependencies

```shell
# Cython needs to be installed before pycocotools
pip3 install cython
pip3 install opencv-python pillow pycocotools matplotlib 
```

If you want to use YOLACT++, compile deformable convolutional layers (from
[DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)).

```shell
cd external/DCNv2
python3 setup.py build develop
```

Modify the configuration file(data/config.py).

```shell
vim data/config.py
# 'train_images': the path of train images
# 'train_info': the path of train_info
# 'valid_images': the path of valid images
# 'valid_info': the path of valid_info
```

## Model Training

By default, we train on COCO. Make sure to download the entire dataset using the commands above.

- To train, grab an imagenet-pretrained model and put it in `./weights`.
  - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
  - For Resnet50, download `resnet50-19c8e357.pth` from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
  - For Darknet53, download `darknet53.pth` from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing).
- Run one of the training commands below.
  - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
  - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.

```shell
# Trains using the base config with a batch size of 8 (the default).
python3 train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python3 train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python3 train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python3 train.py --help
```

YOLACT now supports multiple GPUs seamlessly during training:

- Before running any of the scripts, run: `export CUDA_VISIBLE_DEVICES=[gpus]`
  - Where you should replace [gpus] with a comma separated list of the index of each GPU you want to use (e.g., 0,1,2,3).
  - You should still do this if only using 1 GPU.
  - You can check the indices of your GPUs with `nvidia-smi`.
- Then, simply set the batch size to `8*num_gpus` with the training commands above. The training script will automatically scale the hyperparameters to the right values.
  - If you have memory to spare you can increase the batch size further, but keep it a multiple of the number of GPUs you're using.
  - If you want to allocate the images per GPU specific for different GPUs, you can use `--batch_alloc=[alloc]` where
    [alloc] is a comma seprated list containing the number of images on each GPU. This must sum to `batch_size`.
- The learning rate should divide the number of gpus.

For example: use 8 GPUs to train.

```shell
# Multi-GPU Support
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 train.py --config=yolact_base_config --batch_size 64 --lr 0.000125
```

## Model Results

 | Model  | Result | all   | .50   | .55   | .60   | .65   | .70   | .75   | .80  | .85  | .90  | .95  |
 |--------|--------|-------|-------|-------|-------|-------|-------|-------|------|------|------|------|
 | YOLACT | box    | 11.64 | 23.13 | 21.51 | 19.36 | 16.98 | 14.19 | 10.93 | 6.83 | 2.83 | 0.61 | 0.05 |
 | mask   | 11.20  | 20.51 | 19.07 | 17.51 | 15.52 | 13.23 | 10.74 | 8.10  | 5.13 | 2.06 | 0.13 | 0.13 |

## References

- [yolact](https://github.com/dbolya/yolact)

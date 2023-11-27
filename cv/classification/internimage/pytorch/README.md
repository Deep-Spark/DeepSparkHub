# InternImage for Image Classification

## Model description

"INTERN-2.5" is a powerful multimodal multitask general model jointly released by SenseTime and Shanghai AI Laboratory. It consists of large-scale vision foundation model "InternImage", pre-training method "M3I-Pretraining", generic decoder "Uni-Perceiver" series, and generic encoder for autonomous driving perception "BEVFormer" series.

## Step 1: Installing

### Environment Preparation

-  `CUDA>=10.2` with `cudnn>=7` 
-  `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`

```bash
## Install libGL
yum install -y mesa-libGL

## Install mmcv
cd mmcv/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
cd ../

## Install timm and mmdet
pip3 install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip3 install addict yapf opencv-python termcolor yacs pyyaml scipy
```

- Compiling CUDA operators
```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python3 test.py
cd ../
```

### Data Preparation

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

The ImageNet dataset path structure should look like:

```bash
imagenet
├── train
│   └── n01440764
│       ├── n01440764_10026.JPEG
│       └── ...
├── train_list.txt
├── val
│   └── n01440764
│       ├── ILSVRC2012_val_00000293.JPEG
│       └── ...
└── val_list.txt
```

## Step 2: Training

```bash
# Training on 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export LOCAL_SIZE=8
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py --cfg configs/internimage_t_1k_224.yaml --data-path /path/to/imagenet

# Training on 1 GPU
export CUDA_VISIBLE_DEVICES=0
export LOCAL_SIZE=1
python3 main.py --cfg configs/internimage_t_1k_224.yaml --data-path /path/to/imagenet

```

## Result

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 8 cards     |  Acc@1 83.440     fps 252            |
| 1 card      |                   fps 31             |

## Reference

https://github.com/OpenGVLab/InternImage

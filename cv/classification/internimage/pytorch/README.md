# InternImage

## Model Description

InternImage is a large-scale vision foundation model developed by SenseTime and Shanghai AI Laboratory. It's part of the
INTERN-2.5 multimodal multitask general model, designed for comprehensive visual understanding tasks. The architecture
leverages advanced techniques to achieve state-of-the-art performance in image classification and other vision tasks.
InternImage demonstrates exceptional scalability and efficiency, making it suitable for various applications from
general image recognition to complex autonomous driving perception systems. Its design focuses on balancing model
capacity with computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `/path/to/imagenet` to your ImageNet path in later training process.

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

### Install Dependencies

Environment Preparation.

- `CUDA>=10.2` with `cudnn>=7`
- `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# Install mmcv
cd mmcv/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
cd ../

# Install timm and mmdet
pip3 install timm==0.6.11 mmdet==2.28.1

# Install other requirements:
pip3 install addict yapf opencv-python termcolor yacs pyyaml scipy

# Compiling CUDA operators
cd ./ops_dcnv3
sh ./make.sh

# unit test (should see all checking is True)
python3 test.py

cd ../
```

## Model Training

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

## Model Results

| Model       | GPU        | FP32                     |
|-------------|------------|--------------------------|
| InternImage | BI-V100 x8 | Acc@1 83.440     fps 252 |
| InternImage | BI-V100 x1 | fps 31                   |

## References

- [InternImage](https://github.com/OpenGVLab/InternImage)

# SE_ResNet50_vd

## Model Description

SE_ResNet50_vd is an enhanced version of ResNet50 that incorporates Squeeze-and-Excitation (SE) blocks and variant
downsampling. The SE blocks adaptively recalibrate channel-wise feature responses, improving feature representation. The
variant downsampling preserves more information during feature map reduction. This architecture achieves better accuracy
than standard ResNet50 while maintaining computational efficiency. SE_ResNet50_vd is particularly effective for image
classification tasks, offering improved performance through better feature learning and channel attention mechanisms.

## Model Preparation

### Prepare Resources

Sign up and login in [ImageNet official website](https://www.image-net.org/index.php), then choose 'Download' to
download the whole ImageNet dataset. Specify `./PaddleClas/dataset/` to your ImageNet path in later training process.

The ImageNet dataset path structure should look like:

```bash
ILSVRC2012
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

Tips: for `PaddleClas` training, the image path in train_list.txt and val_list.txt must contain `train/` and `val/`
directories:

- train_list.txt: train/n01440764/n01440764_10026.JPEG 0
- val_list.txt: val/n01667114/ILSVRC2012_val_00000229.JPEG 35

```bash
# add "train/" and "val/" to head of lines
sed -i 's#^#train/#g' train_list.txt
sed -i 's#^#val/#g' val_list.txt
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
python3 -m pip install urllib3==1.26.6
yum install -y mesa-libGL

git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleClas.git
```

## Model Training

```bash
cd PaddleClas/
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch --gpus="0,1,2,3" tools/train.py -c ./ppcls/configs/ImageNet/SENet/SE_ResNet50_vd.yaml
```

## Model Results

| Model          | GPU        | ACC    | FPS              |
|----------------|------------|--------|------------------|
| SE_ResNet50_vd | BI-V100 x8 | 79.20% | 139.63 samples/s |

## References

- [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5)

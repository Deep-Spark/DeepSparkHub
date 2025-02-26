# WSLD: Weighted Soft Label Distillation

## Model description
Knowledge distillation is an effective approach to leverage a well-trained network or an ensemble of them, named as the teacher, to guide the training of a student network. The outputs from the teacher network are used as soft labels for supervising the training of a new network.we investigate the bias-variance tradeoff brought by distillation with soft labels. Specifically, we observe that during training the bias-variance tradeoff varies sample-wisely. Further, under the same distillation temperature setting, we observe that the distillation performance is negatively associated with the number of some specific samples, which are named as regularization samples since these samples lead to bias increasing and variance decreasing. Nevertheless, we empirically find that completely filtering out regularization samples also deteriorates distillation performance. 

## Step 1: Installation
```bash
# Install libGL
yum install mesa-libGL

# Install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/

# Install requirements
pip3 install opencv-python lmdb msgpack
```

## Step 2: Preparing datasets

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

**ImageNet to lmdb file**

The code is used for training Imagenet. Our pre-trained teacher models are Pytorch official models. By default, we pack the ImageNet data as the lmdb file for faster IO. The lmdb files can be made as follows.

```bash
# 1. Generate the list of the image data.
python3 dataset/mk_img_list.py --image_path /path/to/imagenet --output_path /path/to/imagenet

# 2. Use the image list obtained above to make the lmdb file.
python3 dataset/img2lmdb.py --image_path /path/to/imagenet --list_path /path/to/imagenet --output_path '/path/to/imagenet' --split 'train'
python3 dataset/img2lmdb.py --image_path /path/to/imagenet --list_path /path/to/imagenet --output_path '/path/to/imagenet' --split 'val'
```

## Step 3: Training

- train_with_distillation.py: train the model with our distillation method.
- imagenet_train_cfg.py: all dataset and hyperparameter settings.
- knowledge_distiller.py: our weighted soft label distillation loss.

**Hint**: Set up 'data.data_path' to be '/path/to/imagenet' in 'imagenet_train_cfg.py' before you start training.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 train_with_distillation.py
```


## Results

|GPUs|   Network  |  Method  | acc |
|:----:|:----------:|:--------:|:----------:|
| | ResNet 34 |  Teacher | 73.19 |
| | ResNet 18 | Original | 69.75 |
| BI-V100 x 8 | ResNet 18 | Proposed | __71.6__ |

## Reference

- [Rethinking soft labels for knowledge distillation: A Bias-Variance Tradeoff Perspective](https://arxiv.org/abs/2102.00650)
- [Weighted Soft Label Distillation](https://github.com/bellymonster/Weighted-Soft-Label-Distillation)

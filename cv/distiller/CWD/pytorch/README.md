# CWD

> [Channel-wise Knowledge Distillation for Dense Prediction](https://arxiv.org/abs/2011.13256)

<!-- [ALGORITHM] -->

## Model description

Knowledge distillation (KD) has been proven to be a simple and effective tool for training compact models. Almost all KD variants for dense prediction tasks align the student and teacher networks' feature maps in the spatial domain, typically by minimizing point-wise and/or pair-wise discrepancy. Observing that in semantic segmentation, some layers' feature activations of each channel tend to encode saliency of scene categories (analogue to class activation mapping), we propose to align features channel-wise between the student and teacher networks. To this end, we first transform the feature map of each channel into a probability map using softmax normalization, and then minimize the Kullback-Leibler (KL) divergence of the corresponding channels of the two networks. By doing so, our method focuses on mimicking the soft distributions of channels between networks. In particular, the KL divergence enables learning to pay more attention to the most salient regions of the channel-wise maps, presumably corresponding to the most useful signals for semantic segmentation. Experiments demonstrate that our channel-wise distillation outperforms almost all existing spatial distillation methods for semantic segmentation considerably, and requires less computational cost during training. We consistently achieve superior performance on three benchmarks with various network structures.

## Step 1: Installation

```bash
# install libGL
yum install mesa-libGL

# install zlib
wget http://www.zlib.net/fossils/zlib-1.2.9.tar.gz
tar xvf zlib-1.2.9.tar.gz
cd zlib-1.2.9/
./configure && make install
cd ..
rm -rf zlib-1.2.9.tar.gz zlib-1.2.9/

# install requirements
pip3 install cityscapesscripts addict opencv-python

# install mmcv
pushd ../../../../toolbox/MMDetection/patch/mmcv/v2.0.0rc4/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
popd

# install mmrazor
cd ../mmrazor
pip3 install -r requirements.txt
pip3 install mmcls==v1.0.0rc6
pip3 install mmsegmentation==v1.0.0
pip3 install mmengine==0.7.3
python3 setup.py develop 
```

## Step 2: Preparing datasets

Cityscapes 官方网站可以下载 [Cityscapes](<https://www.cityscapes-dataset.com/>) 数据集，按照官网要求注册并登陆后，数据可以在[这里](<https://www.cityscapes-dataset.com/downloads/>)找到。

```bash
mkdir data/
cd data/
```

按照惯例，**labelTrainIds.png 用于 cityscapes 训练。 我们提供了一个基于 cityscapesscripts 的脚本用于生成 **labelTrainIds.png。

```bash
  ├── data
  │   ├── cityscapes
  │   │   ├── leftImg8bit
  │   │   │   ├── train
  │   │   │   ├── val
  │   │   ├── gtFine
  │   │   │   ├── train
  │   │   │   ├── val
```

```bash
cd ..
# --nproc 表示 8 个转换进程，也可以省略。
python3 tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```
## Step 3: Training

```bash
# On single GPU
python3 tools/train.py configs/distill/mmseg/cwd/cwd_logits_pspnet_r101-d8_pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py

# Multiple GPUs on one machine
bash tools/dist_train.sh configs/distill/mmseg/cwd/cwd_logits_pspnet_r101-d8_pspnet_r18-d8_4xb2-80k_cityscapes-512x1024.py 8
```

## Results

|       model       |     GPU     | FP32                                 | 
|-------------------| ----------- | ------------------------------------ |
|   pspnet_r18(student)   | 8 cards     | Miou=  75.32                           |

## Reference
- [mmrazor](https://github.com/open-mmlab/mmrazor)

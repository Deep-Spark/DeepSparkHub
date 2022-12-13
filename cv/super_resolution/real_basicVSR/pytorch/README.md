# RealBasicVSR

## Model description

The diversity and complexity of degradations in real-world video super-resolution (VSR) pose non-trivial challenges in inference and training. First, while long-term propagation leads to improved performance in cases of mild degradations, severe in-the-wild degradations could be exaggerated through propagation, impairing output quality. To balance the tradeoff between detail synthesis and artifact suppression, we found an image pre-cleaning stage indispensable to reduce noises and artifacts prior to propagation. Equipped with a carefully designed cleaning module, our RealBasicVSR outperforms existing methods in both quality and efficiency. Second, real-world VSR models are often trained with diverse degradations to improve generalizability, requiring increased batch size to produce a stable gradient. Inevitably, the increased computational burden results in various problems, including 1) speed-performance tradeoff and 2) batch-length tradeoff. To alleviate the first tradeoff, we propose a stochastic degradation scheme that reduces up to 40% of training time without sacrificing performance. We then analyze different training settings and suggest that employing longer sequences rather than larger batches during training allows more effective uses of temporal information, leading to more stable performance during inference. To facilitate fair comparisons, we propose the new VideoLQ dataset, which contains a large variety of real-world low-quality video sequences containing rich textures and patterns. Our dataset can serve as a common ground for benchmarking. Code, models, and the dataset will be made publicly available.


## Step 1: Installing packages

```shell
sh build_env.sh
```

## Step 2: Preparing datasets

```shell
cd /path/to/modelzoo/official/cv/super_resolution/basicVSR/pytorch

# Download REDS
mkdir -p data/REDS
# Homepage of REDS: https://seungjunnah.github.io/Datasets/reds.html
python3 crop_sub_images.py # cut REDS images into patches for fas

# Download UDM10
cd ..
# Homepage of UDM10: https://www.terabox.com/web/share/link?surl=LMuQCVntRegfZSxn7s3hXw&path=%2Fproject%2Fpfnl
```
## Step 3: Download pretrained weights

```shell
mkdir pretrained && cd pretrained
wget https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth
wget https://download.openmmlab.com/mmediting/restorers/real_basicvsr/realbasicvsr_wogan_c64b20_2x30x8_lr1e-4_300k_reds_20211027-0e2ff207.pth
wget https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
cd ..
```

## Step 3: Training

### Training on single card
```shell
python3 train.py <config file> [training args]   # config file can be found in the configs directory
```

### Mutiple GPUs on one machine
```shell
bash dist_train.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

## Reference
https://github.com/open-mmlab/mmediting

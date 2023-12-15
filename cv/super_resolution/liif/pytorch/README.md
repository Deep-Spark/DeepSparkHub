# LIIF[Learning Continuous Image Representation with Local Implicit Image Function]

## Model description

How to represent an image? While the visual world is presented in a continuous manner, machines store and see the images in a discrete way with 2D arrays of pixels. In this paper, we seek to learn a continuous representation for images. Inspired by the recent progress in 3D reconstruction with implicit neural representation, we propose Local Implicit Image Function (LIIF), which takes an image coordinate and the 2D deep features around the coordinate as inputs, predicts the RGB value at a given coordinate as an output. Since the coordinates are continuous, LIIF can be presented in arbitrary resolution. To generate the continuous representation for images, we train an encoder with LIIF representation via a self-supervised task with super-resolution. The learned continuous representation can be presented in arbitrary resolution even extrapolate to x30 higher resolution, where the training tasks are not provided. We further show that LIIF representation builds a bridge between discrete and continuous representation in 2D, it naturally supports the learning tasks with size-varied image ground-truths and significantly outperforms the method with resizing the ground-truths.

## Step 1: Installing packages

```shell
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```shell
# Download DIV2K 
mkdir -p data/DIV2K
# Home page: https://data.vision.ee.ethz.ch/cvl/DIV2K/

# Download validation samples
mkdir -p data/test
# Home page of Set5: http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html
# Home page of Set14: https://github.com/jbhuang0604/SelfExSR
```

## Step 3: Training

### One single GPU
```shell
python3 train.py <config file> [training args]   # config file can be found in the configs directory
```

### Mutiple GPUs on one machine
```shell
bash dist_train.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```
### Example

```shell
bash dist_train.sh configs/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py 8
```
## Results on BI-V100


| GPUs | FP16  | FPS  |  PSNR |
|------|-------| ---- |  ------------ |
| 1x8  | False | 684  |  26.87         |


## Reference
https://github.com/open-mmlab/mmediting
https://arxiv.org/abs/2012.09161


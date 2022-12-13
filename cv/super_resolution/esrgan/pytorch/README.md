# ESRGAN

## Model description

The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge. The code is available at https://github.com/xinntao/ESRGAN .

## Step 1: Installing packages

```shell
$ pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```shell
$ cd /path/to/modelzoo/cv/super_resolution/esrgan/pytorch

# Download DIV2K 
$ mkdir -p data/DIV2K
$ cd data/DIV2K
# Homepage of DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/

# Download validation samples
$ cd ../..
$ mkdir -p data/test
$ cd data/test
# Home page of Set5: http://people.rennes.inria.fr/Aline.Roumy/results/SR_BMVC12.html
# Home page of Set14: https://github.com/jbhuang0604/SelfExSR

$ python3 get_div2k_anno.py 
```

## Step 3: Training

### One single GPU

```shell
$ python3 train.py <config file> [training args]   # config file can be found in the configs directory
```

### Mutiple GPUs on one machine

```shell
$ bash dist_train.sh <config file> <num_gpus> [training args]    # config file can be found in the configs directory 
```

## Reference
https://github.com/open-mmlab/mmediting

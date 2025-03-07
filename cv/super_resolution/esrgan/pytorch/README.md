# ESRGAN

## Model description

The Super-Resolution Generative Adversarial Network (SRGAN) is a seminal work that is capable of generating realistic textures during single image super-resolution. However, the hallucinated details are often accompanied with unpleasant artifacts. To further enhance the visual quality, we thoroughly study three key components of SRGAN - network architecture, adversarial loss and perceptual loss, and improve each of them to derive an Enhanced SRGAN (ESRGAN). In particular, we introduce the Residual-in-Residual Dense Block (RRDB) without batch normalization as the basic network building unit. Moreover, we borrow the idea from relativistic GAN to let the discriminator predict relative realness instead of the absolute value. Finally, we improve the perceptual loss by using the features before activation, which could provide stronger supervision for brightness consistency and texture recovery. Benefiting from these improvements, the proposed ESRGAN achieves consistently better visual quality with more realistic and natural textures than SRGAN and won the first place in the PIRM2018-SR Challenge. The code is available at https://github.com/xinntao/ESRGAN .

## Step 1: Installing packages

```shell
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/open-mmlab/mmagic.git -b v1.2.0 --depth=1
cd mmagic/
pip3 install -e . -v

sed -i 's/diffusers.models.unet_2d_condition/diffusers.models.unets.unet_2d_condition/g' mmagic/models/editors/vico/vico_utils.py
pip install albumentations
```

## Step 2: Preparing datasets

```shell
# Download DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/ or you can follow this tools/dataset_converters/div2k/README.md
$ mkdir -p data/DIV2K
```

## Step 3: Training

### One single GPU
```shell
python3 tools/train.py configs/esrgan/esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py
```

### Mutiple GPUs on one machine
```shell
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/esrgan/esrgan_psnr-x4c64b23g32_1xb16-1000k_div2k.py 8
```

## Reference
[mmagic](https://github.com/open-mmlab/mmagic)

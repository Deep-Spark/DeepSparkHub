# BiSeNetV2

## Model description

BiSeNet V2 is a two-pathway architecture for real-time semantic segmentation. One pathway is designed to capture the spatial details with wide channels and shallow layers, called Detail Branch. In contrast, the other pathway is introduced to extract the categorical semantics with narrow channels and deep layers, called Semantic Branch. The Semantic Branch simply requires a large receptive field to capture semantic context, while the detail information can be supplied by the Detail Branch. Therefore, the Semantic Branch can be made very lightweight with fewer channels and a fast-downsampling strategy. Both types of feature representation are merged to construct a stronger and more comprehensive feature representation.


## Step 1: Installation

```bash
pip3 install -r requirement

# if libGL.so.1 errors.txt
yum install mesa-libGL -y  
```

## Step 2: Preparing datasets

Download cityscapes from [website](https://www.cityscapes-dataset.com/)

```bash
mv /path/to/leftImg8bit_trainvaltest.zip datasets/cityscapes
mv /path/to/gtFine_trainvaltest.zip datasets/cityscapes
cd datasets/cityscapes
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
```

## Step 3: Training

```bash
bash train.sh 8 configs/bisenetv2_city.py
```

## Results

|  GPUs     | FPS | ss    | ssc   | msf   | mscf  |
| --------- | --- | ----- | ----- | ----- | ----- |
| BiSeNetV2 | 156.81 s/it | 73.42 | 74.67 | 74.99 | 75.71 |

## Reference
[BiSeNet](https://github.com/CoinCheung/BiSeNet/tree/master)


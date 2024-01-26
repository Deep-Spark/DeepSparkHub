# BiSeNet

## Model description

BiSeNet V2 is a two-pathway architecture for real-time semantic segmentation. One pathway is designed to capture the spatial details with wide channels and shallow layers, called Detail Branch. In contrast, the other pathway is introduced to extract the categorical semantics with narrow channels and deep layers, called Semantic Branch. The Semantic Branch simply requires a large receptive field to capture semantic context, while the detail information can be supplied by the Detail Branch. Therefore, the Semantic Branch can be made very lightweight with fewer channels and a fast-downsampling strategy. Both types of feature representation are merged to construct a stronger and more comprehensive feature representation.


## Step 1: Installing packages

```shell
$ pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

Download cityscapes from [website](https://www.cityscapes-dataset.com/)

## Step 3: Training

```
$ bash train.sh {num_gpus} configs/bisenetv2_city.py
```

## Results on BI-V100

|           | ss    | ssc   | msf   | mscf  |
| --------- | ----- | ----- | ----- | ----- |
| bisenetv2 | 74.95 | 75.58 | 76.53 | 77.08 |


* Reference
[BiSeNet](https://github.com/CoinCheung/BiSeNet/tree/master)  

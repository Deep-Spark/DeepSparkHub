# PSANet

## Model description

The point-wise spatial attention network (PSANet) to relax the local neighborhood constraint. 
Each position on the feature map is connected to all the other ones through a self-adaptively learned attention mask.
Moreover, information propagation in bi-direction for scene parsing is enabled.
Information at other positions can be collected to help the prediction of the current position and vice versa, information at the current position can be distributed to assist the prediction of other ones.

## Step 1: Installing

### Install packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training

#### Training on COCO dataset

```shell
bash train_psanet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

## Reference

Ref: https://github.com/ycszen/TorchSeg
Ref: [torchvision](../../torchvision/pytorch/README.md)

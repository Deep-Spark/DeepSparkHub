# RetinaNet

## Model description

The paper proposes a method to convert a deep learning object detector into an equivalent spiking neural network. The aim is to provide a conversion framework that is not constrained to shallow network structures and classification problems as in state-of-the-art conversion libraries. The results show that models of higher complexity, such as the RetinaNet object detector, can be converted with limited loss in performance.

## Step 1: Installing packages

```shell

pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'

```

## Step 2: Training on COCO dataset

Download the [COCO Dataset](https://cocodataset.org/#home) 

### Multiple GPUs on one machine

```shell
bash train_retinanet_r50_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

### Parameters

Ref: [torchvision](../../torchvision/pytorch/README.md)

## Reference

https://github.com/pytorch/vision
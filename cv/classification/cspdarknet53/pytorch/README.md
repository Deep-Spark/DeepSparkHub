# CspDarknet53

## Model description

This is an implementation of CSPDarknet53 in pytorch.

## Step 1: Installing

```bash
pip3 install torchsummary
```

## Step 2: Training

### One single GPU

```bash
export CUDA_VISIBLE_DEVICES=0
python3 train.py --batch-size 64 --epochs 120 --data-path /home/datasets/cv/imagenet
```

### 8 GPUs on one machine
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --batch-size 64 --epochs 120 --data-path /home/datasets/cv/imagenet
```

## Result

| GPU         | FP32                                 |
| ----------- | ------------------------------------ |
| 8 cards     |  Acc@1 76.644     fps 1049           |
| 1 card      |                   fps 148            |

## Reference

https://github.com/WongKinYiu/CrossStagePartialNetworks

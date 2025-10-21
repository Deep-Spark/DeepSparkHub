# Semantic segmentation reference training scripts

This folder contains reference training scripts for semantic segmentation.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

All models have been trained on 8x V100 GPUs.

You must modify the following flags:

`--data-path=/path/to/dataset`

## One Card

### deeplabv3_resnet50

```bash
python3 train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet50 --aux-loss
```

### deeplabv3_resnet101

```bash
python3 train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet101 --aux-loss
```

### deeplabv3_mobilenet_v3_large

```bash
python3 train.py --dataset coco -b 4 --model deeplabv3_mobilenet_v3_large --aux-loss --wd 0.000001
```

## DDP

### deeplabv3_resnet50

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet50 --aux-loss
```

### deeplabv3_resnet101

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --lr 0.02 --dataset coco -b 4 --model deeplabv3_resnet101 --aux-loss
```

### deeplabv3_mobilenet_v3_large

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --dataset coco -b 4 --model deeplabv3_mobilenet_v3_large --aux-loss --wd 0.000001
```

## Arguments

```bash
  --data-path DATA_PATH
                        dataset path
  --dataset DATASET     dataset name, [coco | camvid]
  --model MODEL         model
  --aux-loss            auxiliar loss
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --epochs N            number of total epochs to run
  -j N, --workers N     number of data loading workers (default: 16)
  --lr LR               initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --print-freq PRINT_FREQ
                        print frequency
  --output-dir OUTPUT_DIR
                        path where to save
  --resume RESUME       resume from checkpoint
  --start-epoch N       start epoch
  --test-only           Only test the model
  --pretrained          Use pre-trained models from the modelzoo
  --local_rank LOCAL_RANK
                        Local rank
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
  --amp                 Automatic Mixed Precision training
  --padding-channel     Padding the channels of image to 4
  --nhwc                Use NHWC
  --crop-size CROP_SIZE
  --base-size BASE_SIZE
```

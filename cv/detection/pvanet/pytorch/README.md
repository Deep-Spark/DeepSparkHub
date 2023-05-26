# PVANet

## Model description

In object detection, reducing computational cost is as important as improving accuracy for most practical usages. This paper proposes a novel network structure, which is an order of magnitude lighter than other state-of-the-art networks while maintaining the accuracy. Based on the basic principle of more layers with less channels, this new deep neural network minimizes its redundancy by adopting recent innovations including C.ReLU and Inception structure. We also show that this network can be trained efficiently to achieve solid results on well-known object detection benchmarks: 84.9% and 84.2% mAP on VOC2007 and VOC2012 while the required compute is less than 10% of the recent ResNet-101.


## Step 1: Installing packages

```shell
pip3 install 'scipy' 'matplotlib' 'pycocotools' 'opencv-python' 'easydict' 'tqdm'
```

## Step 2: Preparing COCO dataset

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

```bash
coco2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── ...
├── train2017
│   ├── 000000000009.jpg
│   ├── 000000000025.jpg
│   └── ...
├── val2017
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   └── ...
├── train2017.txt
├── val2017.txt
└── ...
```

## Step 3: Training on COCO dataset

### Multiple GPUs on one machine

```shell
bash train_pvanet_dist.sh --data-path /path/to/coco2017/ --dataset coco
```

### On single GPU

```shell
python3 train.py --data-path /path/to/coco2017/ --dataset coco
```
### Arguments

Ref: [torchvision](../../torchvision/pytorch/README.md)

## Parameters

```
  --data-path DATA_PATH
                        dataset
  --dataset DATASET     dataset
  --device DEVICE       device
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        images per gpu, the total batch size is $NGPU x batch_size
  --epochs N            number of total epochs to run
  -j N, --workers N     number of data loading workers (default: 4)
  --lr LR               initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --lr-scheduler LR_SCHEDULER
                        the lr scheduler (default: multisteplr)
  --lr-step-size LR_STEP_SIZE
                        decrease lr every step-size epochs (multisteplr scheduler only)
  --lr-steps LR_STEPS [LR_STEPS ...]
                        decrease lr every step-size epochs (multisteplr scheduler only)
  --lr-gamma LR_GAMMA   decrease lr by a factor of lr-gamma (multisteplr scheduler only)
  --print-freq PRINT_FREQ
                        print frequency
  --output-dir OUTPUT_DIR
                        path where to save
  --resume RESUME       resume from checkpoint
  --start_epoch START_EPOCH
                        start epoch
  --aspect-ratio-group-factor ASPECT_RATIO_GROUP_FACTOR
  --rpn-score-thresh RPN_SCORE_THRESH
                        rpn score threshold for faster-rcnn
  --trainable-backbone-layers TRAINABLE_BACKBONE_LAYERS
                        number of trainable layers of backbone
  --data-augmentation DATA_AUGMENTATION
                        data augmentation policy (default: hflip)
  --sync-bn             Use sync batch norm
  --test-only           Only test the model
  --pretrained          Use pre-trained models from the modelzoo
  --local_rank LOCAL_RANK
                        Local rank
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
  --nhwc                Use NHWC
  --padding-channel       Padding the channels of image to 4
  --amp                 Automatic Mixed Precision training
  --seed SEED           Random seed
```

## Reference

https://github.com/sanghoon/pytorch_imagenet
https://github.com/pytorch/vision
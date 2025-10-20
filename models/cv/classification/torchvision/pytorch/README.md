# Image Classification reference training scripts

This folder contains reference training scripts for image Classification.
They serve as a log of how to train specific models, as provide baseline
training and evaluation scripts to quickly bootstrap research.

You must modify the following flags:

`--data-path=/path/to/dataset`

## One Card

### resnet50
```
python3 train.py --lr 0.02 --data-path /path/to/imagenet --model resnet50
```


## DDP

### resnet50
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --lr 0.02 --data-path /path/to/imagenet --model resnet50
```


## Arguments

```
  --data-path DATA_PATH dataset
  --model MODEL         model
  --device DEVICE       device
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  --epochs N            number of total epochs to run
  -j N, --workers N     number of data loading workers (default: 4)
  --opt OPT             optimizer
  --lr LR               initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --lr-step-size LR_STEP_SIZE
                        decrease lr every step-size epochs
  --lr-gamma LR_GAMMA   decrease lr by a factor of lr-gamma
  --print-freq PRINT_FREQ
                        print frequency
  --output-dir OUTPUT_DIR
                        path where to save
  --resume RESUME       resume from checkpoint
  --start-epoch N       start epoch
  --cache-dataset       Cache the datasets for quicker initialization. It also serializes the transforms
  --sync-bn             Use sync batch norm
  --test-only           Only test the model
  --pretrained          Use pre-trained models from the modelzoo
  --auto-augment AUTO_AUGMENT
                        auto augment policy (default: None)
  --random-erase RANDOM_ERASE
                        random erasing probability (default: 0.0)
  --dali                Use dali as dataloader
  --local_rank LOCAL_RANK
                        Local rank
  --world-size WORLD_SIZE
                        number of distributed processes
  --dist-url DIST_URL   url used to set up distributed training
  --amp                 Automatic Mixed Precision training
  --nhwc                Use NHWC
  --dali-cpu
  --seed SEED           Random seed
  --crop-size CROP_SIZE
  --base-size BASE_SIZE
```

# FairMOT

## Model description

FairMOT is a model for multi-object tracking which consists of two homogeneous branches to predict pixel-wise objectness scores and re-ID features. The achieved fairness between the tasks is used to achieve high levels of detection and tracking accuracy. The detection branch is implemented in an anchor-free style which estimates object centers and sizes represented as position-aware measurement maps. Similarly, the re-ID branch estimates a re-ID feature for each pixel to characterize the object centered at the pixel. Note that the two branches are completely homogeneous which essentially differs from the previous methods which perform detection and re-ID in a cascaded style. It is also worth noting that FairMOT operates on high-resolution feature maps of strides four while the previous anchor-based methods operate on feature maps of stride 32. The elimination of anchors as well as the use of high-resolution feature maps better aligns re-ID features to object centers which significantly improves the tracking accuracy.

## Step 1: Installing packages

```shell
$ pip3 install -r requirements.txt
```

## Step 2: Preparing data

```shell
$ cd /path/to/modelzoo/official/cv/tracking/fairmot/pytorch

# Download MOT17
$ mkdir -p data/MOT
$ cd data/MOT

```
> download MOT17 dataset
> Baidu NetDisk: [[0]](https://pan.baidu.com/s/1lHa6UagcosRBz-_Y308GvQ)
> Google Drive: [[0]](https://drive.google.com/file/d/1ET-6w12yHNo8DKevOVgK1dBlYs739e_3/view?usp=sharing)
> Original dataset webpage: [MOT-17](https://motchallenge.net/data/MOT17/)

```shell
$ unzip -q MOT17.zip
$ mkdir MOT17/images && mkdir MOT17/labels_with_ids
$ mv ./MOT17/train ./MOT17/images/ && mv ./MOT17/test ./MOT17/images/

$ cd ../../
$ python3 src/gen_labels_17.py
```


### Download Pretrained models

```shell
$ cd /path/to/modelzoo/official/cv/tracking/fairmot/pytorch

# Download ctdet_coco_dla_2x
$ mkdir -p models
$ cd models
```
> download models
> DLA-34 COCO pretrained model: [DLA-34 official](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view)
> HRNetV2 ImageNet pretrained model: [HRNetV2-W18 official](https://1drv.ms/u/s!Aus8VCZ_C_33cMkPimlmClRvmpw)， [HRNetV2-W32 official](https://1drv.ms/u/s!Aus8VCZ_C_33dYBMemi9xOUFR0w)

## Step 3: Training

**The available train scripts are as follows:**

```shell
train_dla34_mot17.sh
train_hrnet18_mot17.sh
train_hrnet32_mot17.sh

```


### On single GPU

```shell
$ GPU_NUMS=1 bash <script> --gpus 0
```
for example, wen train dla34 with MOT17 dataset, batchsize is 18, can use cmd: 
```shell
$ GPU_NUMS=1 bash train_dla34_mot17.sh --gpus 0 --batch_size 18
```

### Multiple GPUs on one machine

```shell
$ GPU_NUMS=<gpu numbers> bash <script> --gpus <gpu ids>
```
for example, wen train dla34 with MOT17 dataset, using gpu x8, batchsize is 144(per gpu batchsize is 18), can use cmd: 
```shell
$ GPU_NUMS=8 bash train_dla34_mot17.sh --gpus 0,1,2,3,4,5,6,7 --batch_size 144
```

** To reduce training time span, you can append "--num_epochs 1 --num_iters 300" to the command. **

### Training arguments

```python

# dataloader threads. 0 for single-thread.
num_workers: int = 8

# gpus model trained on, -1 for CPU, use comma for multiple gpus
# gpu is indexed starting from 0
# When multiple gpus are used, gpus of consecutive index are allowed by default. If you want to use gpus of nonconsecutive index, set CUDA_VISIBLE_DEVICES environment variable.
gpus: default = '0'

# not use torch.backends.cudnn.benchmark = True
# If needed, use --not_cuda_benchmark on command line.
not_cuda_benchmark: action = 'store_true'

# batch size of all gpus
batch_size: int = 12

# total training epochs
num_epochs: int = 30

# number of iters in one epoch,set to -1 use (samples_number / batch_size)
num_iters: int = -1

```

## Testing
> Only command on testing dataset is provided. MOT dataset is not public. To access MOT, submit via [motchallenge](https://motchallenge.net/instructions/).

```shell
cd /path/to/modelzoo/official/cv/tracking/fairmot/pytorch/src
python3 track.py mot --val_mot17 True --load_model /path/to/saved/model --conf_thres 0.4 --data_dir ../data/MOT

```

## Results on BI-V100

| GPUs | FPS   | MOTA |
|------|-------| ------------ |
| 1x8  | 28.5 | 69.8         |


| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
|----------------------|------------------------------------------|-------------|----------|------------|-------------|-------------------------|-----------|
| MOTA:69.8            | SDK V2.2,bs:64,8x,fp32                   | 52          | 69.8     | 132\*8     | 0.97        | 19.1\*8                 | 1         |


## Reference
https://github.com/ifzhang/FairMOT

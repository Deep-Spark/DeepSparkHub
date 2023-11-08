# BEVFormer

## Model description
In this work, the authors present a new framework termed BEVFormer, which learns unified BEV representations with spatiotemporal transformers to support multiple autonomous driving perception tasks. In a nutshell, BEVFormer exploits both spatial and temporal information by interacting with spatial and temporal space through predefined grid-shaped BEV queries. To aggregate spatial information, the authors design a spatial cross-attention that each BEV query extracts the spatial features from the regions of interest across camera views. For temporal information, the authors propose a temporal self-attention to recurrently fuse the history BEV information.
The proposed approach achieves the new state-of-the-art **56.9\%** in terms of NDS metric on the nuScenes test set, which is **9.0** points higher than previous best arts and on par with the performance of LiDAR-based baselines.


## Prepare
**Install mmcv-full.**
```shell
cd mmcv
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh
```

**Install mmdet and mmseg.**
```shell
pip3 install mmdet==2.25.0
pip3 install mmsegmentation==0.25.0
```

**Install mmdet3d from source code.**
```shell
cd ../mmdetection3d
pip3 install -r requirements.txt,OR pip3 install -r requirements/optional.txt,pip3 install -r requirements/runtime.txt,pip3 install -r requirements/tests.txt
python3 setup.py install
```

**Install timm.**
```shell
pip3 install timm
```

## NuScenes
Download nuScenes V1.0-mini data and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**
```
cd ..
mkdir data
cd data  
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*
```
cd ..
python3 tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

## Prepare pretrained models
```shell
mkdir ckpts
cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
cd ..
```

## Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

## Train and Test

Train BEVFormer with 8 GPUs 
```
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8
```

Eval BEVFormer with 8 GPUs
```
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8
```
Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple GPUs. By default we report the score evaled with 8 GPUs.



## Using FP16 to train the model.
The above training script can not support FP16 training, 
and we provide another script to train BEVFormer with FP16.

```
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```
## Results on BI-V100

| GPUs |     model      |   NDS  |   mAP  |
|------|----------------|--------|--------|
| 1x8  | bevformer_base | 0.3516 | 0.3701 |

## Reference:
[Paper in arXiv](http://arxiv.org/abs/2203.17270) 

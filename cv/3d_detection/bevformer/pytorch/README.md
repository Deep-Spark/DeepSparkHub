# BEVFormer

## Model Description

BEVFormer is a transformer-based framework for autonomous driving perception that learns unified Bird's Eye View (BEV)
representations. It combines spatial and temporal information through innovative attention mechanisms: spatial
cross-attention extracts features from camera views, while temporal self-attention fuses historical BEV data. This
approach achieves state-of-the-art performance on nuScenes dataset, matching LiDAR-based systems. BEVFormer supports
multiple perception tasks simultaneously, making it a versatile solution for comprehensive scene understanding in
autonomous driving applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Download nuScenes V1.0-mini data and CAN bus expansion data from [HERE](https://www.nuscenes.org/download). Prepare
nuscenes data by running.

```bash
mkdir data
cd data/

# download 'can_bus.zip'
unzip can_bus.zip 

# move can_bus to data dir
```

Prepare nuScenes data.

We genetate custom annotation files which are different from mmdet3d's

```bash
cd ../
python3 tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
```

Using the above code will generate `nuscenes_infos_temporal_{train,val}.pkl`.

Prepare pretrained models.

```shell
mkdir ckpts
cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth
cd ../
```

### Install Dependencies

```shell
# Install mmcv-full
cd mmcv/
bash clean_mmcv.sh
bash build_mmcv.sh
bash install_mmcv.sh

# Install mmdet and mmseg
pip3 install mmdet==2.25.0
pip3 install mmsegmentation==0.25.0

# Install mmdet3d from source code
cd ../mmdetection3d
pip3 install -r requirements.txt,OR pip3 install -r requirements/optional.txt,pip3 install -r requirements/runtime.txt,pip3 install -r requirements/tests.txt
python3 setup.py install

# Install timm
pip3 install timm
```

## Model Training

```bash
# Train BEVFormer with 8 GPUs
./tools/dist_train.sh ./projects/configs/bevformer/bevformer_base.py 8

# Eval BEVFormer with 8 GPUs
./tools/dist_test.sh ./projects/configs/bevformer/bevformer_base.py ./path/to/ckpts.pth 8
```

Note: using 1 GPU to eval can obtain slightly higher performance because continuous video may be truncated with multiple
GPUs. By default we report the score evaled with 8 GPUs.

The above training script can not support FP16 training,
and we provide another script to train BEVFormer with FP16.

```bash
# Using FP16 to train the model
./tools/fp16/dist_train.sh ./projects/configs/bevformer_fp16/bevformer_tiny_fp16.py 8
```

## Model Results

| Model     | GPU        | model          | NDS    | mAP    |
|-----------|------------|----------------|--------|--------|
| BEVFormer | BI-V100 x8 | bevformer_base | 0.3516 | 0.3701 |

## References

[BEVFormer](https://github.com/fundamentalvision/BEVFormer/tree/master)

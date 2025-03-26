# BasicVSR++

## Model Description

BasicVSR++ is an advanced video super-resolution model that enhances BasicVSR with second-order grid propagation and
flow-guided deformable alignment. These innovations improve spatiotemporal information utilization across misaligned
frames, boosting performance by 0.82 dB PSNR while maintaining efficiency. It excels in video restoration tasks,
including compressed video enhancement, and achieved top results in NTIRE 2021 challenges. The model's recurrent
structure effectively processes entire video sequences, making it a state-of-the-art solution for high-quality video
upscaling and restoration.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Download REDS dataset from [homepage](https://seungjunnah.github.io/Datasets/reds.html) or you can follow
tools/dataset_converters/reds/README.md

```shell
mkdir -p data/
ln -s ${REDS_DATASET_PATH} data/REDS
```

### Install Dependencies

```shell
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

git clone https://github.com/open-mmlab/mmagic.git -b v1.2.0 --depth=1
cd mmagic/
pip3 install -e . -v

sed -i 's/diffusers.models.unet_2d_condition/diffusers.models.unets.unet_2d_condition/g' mmagic/models/editors/vico/vico_utils.py
pip install albumentations
```

## Model Training

```shell
# One single GPU
python3 tools/train.py configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py

# Mutiple GPUs on one machine
sed -i 's/python /python3 /g' tools/dist_train.sh
bash tools/dist_train.sh configs/basicvsr_pp/basicvsr-pp_c64n7_8xb1-600k_reds4.py 8
```

## References

- [mmagic](https://github.com/open-mmlab/mmagic)

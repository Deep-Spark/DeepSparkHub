# CenterNet

## Model Description

CenterNet is an efficient object detection model that represents objects as single points (their bounding box centers)
rather than traditional bounding boxes. It uses keypoint estimation to locate centers and regresses other object
properties like size and orientation. This approach eliminates the need for anchor boxes and non-maximum suppression,
making it simpler and faster. CenterNet achieves state-of-the-art speed-accuracy trade-offs on benchmarks like COCO and
can be extended to 3D detection and pose estimation tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

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

```bash
ln -s /path/to/coco2017 ./data/coco
```

```bash
# Prepare offline file "resnet18-5c106cde.pth" if download fails
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
mkdir -p /root/.cache/torch/hub/checkpoints/
mv resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/
```

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

pip3 install -r requirements.txt
git clone https://github.com/xingyizhou/CenterNet.git
git checkout 4c50fd3a46bdf63dbf2082c5cbb3458d39579e6c

# Compile deformable convolutional(DCNv2)
cd ./src/lib/models/networks/
rm -rf DCNv2
git clone -b pytorch_1.11 https://github.com/lbin/DCNv2.git
cd ./DCNv2/
python3 setup.py build develop
```

## Model Training

```bash
cd ./src
touch lib/datasets/__init__.py
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# On single GPU
python3 main.py ctdet --arch res_18 --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0

# Multiple GPUs on one machine
python3 main.py ctdet --arch res_18 --batch_size 128 --master_batch 60 --lr 1.25e-4  --gpus 0,1,2,3,4,5,6,7
```

## References

- [CenterNet](https://github.com/xingyizhou/CenterNet)

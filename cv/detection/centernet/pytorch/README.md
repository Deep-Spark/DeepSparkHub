# CenterNet

## Model description

Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

## Step 1: Installing packages

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

## Step 2: Preparing datasets

### Go back to the "pytorch/" directory

```bash
cd ../../../../../
```

### Download coco2017

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

### Set up soft link to coco2017

```bash
ln -s /path/to/coco2017 ./data/coco
```

### Prepare offline file "resnet18-5c106cde.pth" if download fails

```bash
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
mkdir -p /root/.cache/torch/hub/checkpoints/
mv resnet18-5c106cde.pth /root/.cache/torch/hub/checkpoints/
```

## Step 3: Training

### Setup CUDA_VISIBLE_DEVICES variable

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### On single GPU

```bash
cd ./cv/detection/centernet/pytorch/src
touch lib/datasets/__init__.py
python3 main.py ctdet --arch res_18 --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0
```

### Multiple GPUs on one machine

```bash
python3 main.py ctdet --arch res_18 --batch_size 128 --master_batch 60 --lr 1.25e-4  --gpus 0,1,2,3,4,5,6,7
```

## Reference
https://github.com/xingyizhou/CenterNet

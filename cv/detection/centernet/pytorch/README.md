# CenterNet

## Model description
Detection identifies objects as axis-aligned boxes in an image. Most successful object detectors enumerate a nearly exhaustive list of potential object locations and classify each. This is wasteful, inefficient, and requires additional post-processing. In this paper, we take a different approach. We model an object as a single point --- the center point of its bounding box. Our detector uses keypoint estimation to find center points and regresses to all other object properties, such as size, 3D location, orientation, and even pose. Our center point based approach, CenterNet, is end-to-end differentiable, simpler, faster, and more accurate than corresponding bounding box based detectors. CenterNet achieves the best speed-accuracy trade-off on the MS COCO dataset, with 28.1% AP at 142 FPS, 37.4% AP at 52 FPS, and 45.1% AP with multi-scale testing at 1.4 FPS. We use the same approach to estimate 3D bounding box in the KITTI benchmark and human pose on the COCO keypoint dataset. Our method performs competitively with sophisticated multi-stage methods and runs in real-time.

## Step 1: Installing packages

```bash
pip3 install -r requirements.txt
# Compile deformable convolutional(DCNv2)
cd ./src/lib/models/networks/
git clone -b pytorch_1.9 https://github.com/lbin/DCNv2.git
cd ./DCNv2/
python3 setup.py build develop

```

## Step 2: Preparing datasets

```bash
# Go back to the "pytorch/" directory
cd ../../../../../

# Download from homepage of coco: https://cocodataset.org/
ln -s ${coco_2017_dataset_path} ./data/coco

# The "data/coco" directory would be look like
data/coco
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017
├── train2017.txt
├── val2017
└── val2017.txt

# Prepare offline file "resnet18-5c106cde.pth" if download fails
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
python3 main.py ctdet --batch_size 32 --master_batch 15 --lr 1.25e-4  --gpus 0
```

### Multiple GPUs on one machine
```bash
cd ./cv/detection/centernet/pytorch/src
bash run.sh
```

## Reference
https://github.com/bubbliiiing/centernet-pytorch

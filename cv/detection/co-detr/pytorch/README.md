# Co-DETR (DETRs with Collaborative Hybrid Assignments Training)
This repo is the official implementation of ["DETRs with Collaborative Hybrid Assignments Training"](https://arxiv.org/pdf/2211.12860.pdf) by Zhuofan Zong, Guanglu Song, and Yu Liu.

## Model description

In this paper, we present a novel collaborative hybrid assignments training scheme, namely Co-DETR, to learn more efficient and effective DETR-based detectors from versatile label assignment manners. 
1. **Encoder optimization**: The proposed training scheme can easily enhance the encoder's learning ability in end-to-end detectors by training multiple parallel auxiliary heads supervised by one-to-many label assignments. 
2. **Decoder optimization**: We conduct extra customized positive queries by extracting the positive coordinates from these auxiliary heads to improve attention learning of the decoder. 
3. **State-of-the-art performance**: Co-DETR with [ViT-L](https://github.com/baaivision/EVA/tree/master/EVA-02) (304M parameters) is **the first model to achieve 66.0 AP on COCO test-dev.**

## Step 1: Installation
### (1) install MMCV
```bash
# Go to "toolbox/MMDetection" directory in root path
bash install_toolbox_mmdetection.sh
```
### (2) install other
```bash
pip3 install -r requirements.txt
pip3 install urllib3==1.26.15
yum install -y mesa-libGL
```

## Step 2: Preparing datasets

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to download.

Take coco2017 dataset as an example, specify `/path/to/coco` to your COCO path in later training process, the unzipped dataset path structure sholud look like:

```bash
coco
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
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

## Step 3: Training

```bash
# Make coco dataset path soft link to ./data/coco
mkdir data/
ln -s /path/to/coco ./data
```

```bash
# One GPU
export CUDA_VISIBLE_DEVICES=0
python3 train.py projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py --work-dir path_to_exp --no-validate --auto-resume

# Eight GPUs
bash tools/dist_train.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py 8 path_to_exp --no-validate --auto-resume

# Evaluation
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=".:$PYTHONPATH" python3 tools/test.py projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_exp/latest.pth --eval bbox
```

## Results

| GPUs | FPS | Train Epochs | Box AP |
|------|---------|----------|--------|
| BI-V100 x8 | 9.02 | 12  | 0.428    |

## Reference
- [Co-DETR](https://github.com/Sense-X/Co-DETR)
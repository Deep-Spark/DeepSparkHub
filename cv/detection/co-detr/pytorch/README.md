# Co-DETR (DETRs with Collaborative Hybrid Assignments Training)

## Model Description

Co-DETR is an advanced object detection model that enhances DETR (DEtection TRansformer) through collaborative hybrid
assignments training. It improves encoder learning by training multiple auxiliary heads with one-to-many label
assignments and optimizes decoder attention using customized positive queries. Co-DETR achieves state-of-the-art
performance, being the first model to reach 66.0 AP on COCO test-dev with ViT-L. This approach significantly boosts
detection accuracy and efficiency while maintaining end-to-end training simplicity.

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco` to your COCO path in later training process, the unzipped
dataset path structure sholud look like:

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

### Install Dependencies

```bash
# Install libGL
## CentOS
yum install -y mesa-libGL
## Ubuntu
apt install -y libgl1-mesa-glx

# install MMDetection
git clone https://github.com/open-mmlab/mmdetection.git -b v3.3.0 --depth=1
cd mmdetection
pip install -v -e .

# Install requirements
pip3 install -r requirements.txt
pip3 install urllib3==1.26.15

# Download repo
git clone https://github.com/Sense-X/Co-DETR.git
cd /path/to/Co-DETR
git checkout bf3d49d7c02929788dfe2f251b6b01cbe196b736
```

## Model Training

```bash
# Make coco dataset path soft link to ./data/coco
mkdir data/
ln -s /path/to/coco ./data

# One GPU
export CUDA_VISIBLE_DEVICES=0
python3 tools/train.py projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py --work-dir path_to_exp --no-validate --auto-resume

# Eight GPUs
bash tools/dist_train.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py 8 path_to_exp --no-validate --auto-resume

# Evaluation
export CUDA_VISIBLE_DEVICES=0
PYTHONPATH=".:$PYTHONPATH" python3 tools/test.py projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_exp/latest.pth --eval bbox
```

## Model Results

| Model   | GPU        | FPS  | Train Epochs | Box AP |
|---------|------------|------|--------------|--------|
| Co-DETR | BI-V100 x8 | 9.02 | 12           | 0.428  |

## References

- [Paper](https://arxiv.org/pdf/2211.12860.pdf)
- [Co-DETR](https://github.com/Sense-X/Co-DETR)

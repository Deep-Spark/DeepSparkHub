# SSD

## Model Description

SSD (Single Shot MultiBox Detector) is a fast and efficient object detection model that predicts bounding boxes and
class scores in a single forward pass. It uses a set of default boxes at different scales and aspect ratios across
multiple feature maps to detect objects of various sizes. SSD combines predictions from different layers to handle
objects at different resolutions, offering a good balance between speed and accuracy for real-time detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

Go to visit [COCO official website](https://cocodataset.org/#download), then select the COCO dataset you want to
download.

Take coco2017 dataset as an example, specify `/path/to/coco2017` to your COCO path in later training process, the
unzipped dataset path structure sholud look like:

```bash
mkdir -p data/datasets/

data/datasets/coco2017
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

### Install Dependencies
Contact the Iluvatar administrator to get the missing packages:
    - dali-1.21.0+corex.4.3.0-cp310-cp310-linux_x86_64.whl
    - apex-0.1+corex.4.3.0-cp310-cp310-linux_x86_64.whl

```bash
apt install -y git numactl
pip3 install "git+https://github.com/mlperf/logging.git@1.0-branch" pybind11==2.9.2 ujson==1.35
pip3 install wheel numpy>=1.26.4 cython pycocotools==2.0.8

bash ./clean_ssd.sh && bash ./build_ssd.sh && bash ./install_ssd.sh "$@"
export DATA_PATH_BBOX=../../../..
export DATA_PATH=data/datasets/coco2017
python3 prepare-json.py --keep-keys ${DATA_PATH}/annotations/instances_val2017.json ${DATA_PATH_BBOX}/bbox_only_instances_val2017.json "$@"
python3 prepare-json.py ${DATA_PATH}/annotations/instances_train2017.json ${DATA_PATH_BBOX}/bbox_only_instances_train2017.json "$@"
```

## Model Training

```bash
python3 train.py --dali --dali-cache 0 --data=${DATA_PATH} \
--batch-size=160 --warmup-factor=0 --warmup=650 --lr=2.92e-3 --threshold=0.08 --epochs 5 --eval-batch-size=160 \
--wd=1.6e-4 --use-fp16 --jit --nhwc --pad-input --delay-allreduce --lr-decay-factor=0.2 --lr-decay-epochs 34 45 --opt-level O2 --seed 1769250163 "$@"
```

## Model Results

| Model | GPU        | Batch Size | IoU=0.50:0.95  | IoU=0.50 | IoU=0.75  |
|-------|------------|------------|------|--------------|------|
| SSD   | BI-V150 x8 | 160        | 0.094 | 0.197           | 0.078 |

## References

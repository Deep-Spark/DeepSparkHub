# FCOS

## Model Description

FCOS (Fully Convolutional One-Stage Object Detection) is an anchor-free object detection model that predicts bounding
boxes directly without anchor boxes. It uses a fully convolutional network to detect objects by predicting per-pixel
bounding boxes and class labels. FCOS simplifies the detection pipeline, reduces hyperparameters, and achieves
competitive performance on benchmarks like COCO. Its center-ness branch helps suppress low-quality predictions, making
it efficient and effective for various detection tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

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
mkdir -p data
ln -s /path/to/coco2017 data/coco

wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
mv R-50.pkl /root/.torch/models/
```

### Install Dependencies

```bash
pip3 install -r requirements.txt
python3 setup.py develop
```

## Model Training

The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent
(SGD):

```bash
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    tools/train_net.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x
```

Note that:

1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be
   changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size,
   please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) If you want to train FCOS on your own dataset, please follow this instruction
   [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
5) Now, training with 8 GPUs and 4 GPUs can have the same performance. Previous performance gap was because we did not
   synchronize `num_pos` between GPUs when computing loss.

## Model Results

 | Model | GPU        | FPS  | Train Epochs | Box AP |
 |-------|------------|------|--------------|--------|
 | FCOS  | BI-V100 x8 | 8.24 | 12           | 38.7   |

## References

- [FCOS](https://github.com/tianzhi0549/FCOS)

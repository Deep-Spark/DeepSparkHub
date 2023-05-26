# FCOS

## Model description
FCOS (Fully Convolutional One-Stage Object Detection) is a fast anchor-free object detection framework with strong performance.
The full paper is available at: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355). 

## Install requirements

```
pip3 install -r requirements.txt
python3 setup.py develop
```

## Prepare datasets

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

```bash
mkdir -p data
ln -s /path/to/coco2017 data/coco

wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
mv R-50.pkl /root/.torch/models/
```

## Training

The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python3 -m torch.distributed.launch \
        --nproc_per_node=8 \
        tools/train_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x
        
Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
5) Now, training with 8 GPUs and 4 GPUs can have the same performance. Previous performance gap was because we did not synchronize `num_pos` between GPUs when computing loss. 

## Results on BI-V100

| GPUs | FPS | Train Epochs | Box AP|
|------|-----|--------------|-------|
| 1x8  | 8.24 | 12          |  38.7 |

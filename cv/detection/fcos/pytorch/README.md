# FCOS: Fully Convolutional One-Stage Object Detection
## Model description
FCOS (Fully Convolutional One-Stage Object Detection) is a fast anchor-free object detection framework with strong performance.
The full paper is available at: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355). 

## install

```
pip3 install -r requirements.txt
python3 setup.py develop
```

## datasets
```
mkdir data
cd data
ln -s /home/datasets/cv/coco2017 coco
cd ..
wget https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/MSRA/R-50.pkl
mv R-50.pkl /root/.torch/models/
```

## Training

The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python3 -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
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

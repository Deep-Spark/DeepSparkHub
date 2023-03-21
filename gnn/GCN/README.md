# GCN

## Model description
GCN(Graph Convolutional Networks) was proposed in 2016 and designed to do semi-supervised learning on graph-structured data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations that encode both local graph structure and features of nodes.

[Paper](https://arxiv.org/abs/1609.02907):  Thomas N. Kipf, Max Welling. 2016. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR 2016.

## Step 1: Installing
```
pip3 install -r requirements.txt
pip3 install easydict
```

## Step 2: Prepare Datasets

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

| Dataset  | Type             | Nodes | Edges | Classes | Features | Label rate |
| -------  | ---------------: |-----: | ----: | ------: |--------: | ---------: |
| Cora    | Citation network | 2708  | 5429  | 7       | 1433     | 0.052      |
| Citeseer| Citation network | 3327  | 4732  | 6       | 3703     | 0.036      |

## Step 3: Training
```
cd scripts 
bash train_gcn_1p.sh
```
## Evaluation

```bash
cd ..
python3 eval.py --data_dir=scripts/data_mr/cora --device_target="GPU" --model_ckpt scripts/train/ckpt/ckpt_gcn-200_1.ckpt  &> eval.log &
```

## Evaluation result

### Results on BI-V100

| GPUs | per step time  |  Acc  |
|------|--------------  |-------|
|   1  |   4.454        | 0.8711|

### Results on NV-V100s

| GPUs | per step time  |  Acc  |
|------|--------------  |-------|
|   1  |   5.278        | 0.8659|

# GCN

## Model Description

GCN (Graph Convolutional Networks) was proposed in 2016 and designed to do semi-supervised learning on graph-structured
data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on
graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations
that encode both local graph structure and features of nodes.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant
domain/network architecture. In the following sections, we will introduce how to run the scripts using the related
dataset below.

| Dataset  | Type             | Nodes | Edges | Classes | Features | Label rate |
|----------|------------------|-------|-------|---------|----------|------------|
| Cora     | Citation network | 2708  | 5429  | 7       | 1433     | 0.052      |
| Citeseer | Citation network | 3327  | 4732  | 6       | 3703     | 0.036      |

### Install Dependencies

```sh
pip3 install -r requirements.txt
pip3 install easydict
```

## Model Training

```sh
cd scripts/
bash train_gcn_1p.sh

# Evaluation
cd ../
python3 eval.py --data_dir=scripts/data_mr/cora --device_target="GPU" \
                --model_ckpt scripts/train/ckpt/ckpt_gcn-200_1.ckpt &> eval.log &
```

## Model Results

| Model | GPU         | per step time | Acc    |
|-------|-------------|---------------|--------|
| GCN   | BI-V100 x1  | 4.454         | 0.8711 |
| GCN   | NV-V100s x1 | 5.278         | 0.8659 |

## References

- [Paper](https://arxiv.org/abs/1609.02907)

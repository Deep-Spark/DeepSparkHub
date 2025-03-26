# GAT

## Model Description

GAT (Graph Attention Network) is a novel neural network architecture for graph-structured data that uses self-attention
mechanisms to process node features. Unlike traditional graph convolutional networks, GAT assigns different importance
to neighboring nodes through attention coefficients, allowing for more flexible and expressive feature aggregation. This
approach enables the model to handle varying neighborhood sizes and capture complex relationships in graph data, making
it particularly effective for tasks like node classification and graph-based prediction problems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

There's no need to prepare dastasets. The details for datasets can be found in the
[paper](https://arxiv.org/abs/1609.02907).

### Install Dependencies

```bash
git clone -b 2.2.5 https://github.com/PaddlePaddle/PGL
pip3 install pgl==2.2.5
```

## Model Training

```bash
cd PGL/examples/gat/
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset cora
```

## Model Results

| Model | GPU     | Accuracy | FPS        |
|-------|---------|----------|------------|
| GAT   | BI-V100 | 83.16%   | 65.56 it/s |

## References

- [PGL](https://github.com/PaddlePaddle/PGL)

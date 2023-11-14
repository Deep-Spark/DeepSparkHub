# GAT (Graph Attention Networks)

[Graph Attention Networks \(GAT\)](https://arxiv.org/abs/1710.10903) is a novel architectures that operate on graph-structured data, which leverages masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. Based on PGL, we reproduce GAT algorithms and reach the same level of indicators as the paper in citation network benchmarks.

## Step 1: Installation

```bash
git clone -b 2.2.5 https://github.com/PaddlePaddle/PGL
pip3 install pgl==2.2.5
```

## Step 2: Preparing datasets

There's no need to prepare dastasets. The details for datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

## Step 3: Training

```bash
cd PGL/examples/gat/
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset cora
```

## Results

| GPUs | Accuracy | FPS |
| --- | --- | --- |
| BI-V100 | 83.16% | 65.56 it/s |

## Reference

- [PGL](https://github.com/PaddlePaddle/PGL)

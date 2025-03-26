# GCN

## Model Description

GCN (Graph Convolutional Networks) was proposed in 2016 and designed to do semi-supervised learning on graph-structured
data. A scalable approach based on an efficient variant of convolutional neural networks which operate directly on
graphs was presented. The model scales linearly in the number of graph edges and learns hidden layer representations
that encode both local graph structure and features of nodes.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

Datasets are called in the code.

The datasets contain three citation networks: CORA, PUBMED, CITESEER.

### Install Dependencies

```sh
# Clone PGL repository
git clone https://github.com/PaddlePaddle/PGL.git
```

```sh
# Pip the requirements
pip3 install pgl
pip3 install urllib3==1.23
pip3 install networkx
```

## Model Training

```sh
cd PGL/examples/gcn/

# Run on GPU
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataset cora
```

## Model Results

 | Model | GPU        | Datasets | speed  | Accurary |
 |-------|------------|----------|--------|----------|
 | GCN   | BI-V100 ×1 | CORA     | 0.0064 | 80.3%    |
 | GCN   | BI-V100 ×1 | PUBMED   | 0.0076 | 79.0%    |
 | GCN   | BI-V100 ×1 | CITESEER | 0.0085 | 70.6%    |

## References

- [PGL](https://github.com/PaddlePaddle/PGL/tree/main/examples/gcn)
- [Paper](https://arxiv.org/abs/1609.02907)

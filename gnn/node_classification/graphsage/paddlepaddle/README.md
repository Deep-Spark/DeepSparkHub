# GraphSAGE (Inductive Representation Learning on Large Graphs)

[GraphSAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) is a general inductive framework that leverages node feature
information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, GraphSAGE learns a function that generates embeddings by sampling and aggregating features from a node’s local neighborhood. Based on PGL, we reproduce GraphSAGE algorithm and reach the same level of indicators as the paper in Reddit Dataset. Besides, this is an example of subgraph sampling and training in PGL.

## Step 1: Installation

```bash
git clone -b 2.2.5 https://github.com/PaddlePaddle/PGL
pip3 install scikit-learn
pip3 install pgl==2.2.5
```

## Step 2: Preparing datasets
The reddit dataset should be downloaded from the following links and placed in the directory ```pgl.data```. The details for Reddit Dataset can be found [here](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).

- reddit.npz https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J
- reddit_adj.npz: https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt

```bash
# Make soft link to reddit dataset path
ln -s /path/to/reddit/ /usr/local/lib/python3.7/site-packages/pgl/data/
```

## Step 3: Training

T  train a GraphSAGE model on Reddit Dataset, you can just run

```bash
cd PGL/examples/graphsage/cpu_sample_version

CUDA_VISIBLE_DEVICES=0 python3 train.py  --epoch 10  --normalize --symmetry
```

## Results

| GPUs | Accuracy   | FPS |
| --- | --- | --- |
| BI-V100 x1 | 0.9072 | 47.54 s/it |

## Reference

- [PGL](https://github.com/PaddlePaddle/pgl)

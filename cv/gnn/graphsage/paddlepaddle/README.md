# GraphSAGE

## Model Description

GraphSAGE (Graph Sample and Aggregated) is an inductive graph neural network model designed for large-scale graph data.
Unlike traditional methods that learn fixed node embeddings, GraphSAGE learns a function to generate embeddings by
sampling and aggregating features from a node's local neighborhood. This approach enables the model to generalize to
unseen nodes and graphs, making it particularly effective for dynamic graphs and large-scale applications like social
network analysis and recommendation systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

The reddit dataset should be downloaded from the following links and placed in the directory ```pgl.data```. The details
for Reddit Dataset can be found [here](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf).

- reddit.npz <https://drive.google.com/open?id=19SphVl_Oe8SJ1r87Hr5a6znx3nJu1F2J>
- reddit_adj.npz: <https://drive.google.com/open?id=174vb0Ws7Vxk_QTUtxqTgDHSQ4El4qDHt>

```sh
# Make soft link to reddit dataset path
ln -s /path/to/reddit/ /usr/local/lib/python3.7/site-packages/pgl/data/
```

### Install Dependencies

```sh
git clone -b 2.2.5 https://github.com/PaddlePaddle/PGL
pip3 install scikit-learn
pip3 install pgl==2.2.5
```

## Model Training

To  train a GraphSAGE model on Reddit Dataset, you can just run:

```sh
cd PGL/examples/graphsage/cpu_sample_version

CUDA_VISIBLE_DEVICES=0 python3 train.py  --epoch 10  --normalize --symmetry
```

## Model Results

| GPUs       | Accuracy | FPS        |
|------------|----------|------------|
| BI-V100 x1 | 0.9072   | 47.54 s/it |

## References

- [PGL](https://github.com/PaddlePaddle/pgl)

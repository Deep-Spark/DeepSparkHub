# NCF

## Model Description

NCF (Neural Collaborative Filtering) is an advanced recommendation system model that replaces traditional matrix
factorization with neural networks. It learns user-item interactions through a multi-layer perceptron, enabling it to
capture complex, non-linear relationships. NCF generalizes matrix factorization while offering enhanced flexibility and
performance. It significantly improves recommendation accuracy by leveraging deep learning capabilities. The model is
particularly effective for collaborative filtering tasks, demonstrating superior results on real-world datasets compared
to traditional methods. NCF's architecture makes it a powerful tool for personalized recommendation systems.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Dataset is movielens.

```sh
# Download dataset
mkdir -p data/
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip -P data/

# Unzip
unzip data/ml-20m.zip -d data/

# Convert
python3 convert.py --path ./data/ml-20m/ratings.csv --output ./data/ml-20m
```

### Install Dependencies

```sh
pip3 install -r requirements.txt
```

## Model Training

```sh
# Multiple GPUs on one machine
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
bash run_train_fp32.sh

# Multiple GPUs on one machine (AMP)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
## fp16 train
bash run_train_fp16.sh
```

## References

- [mlcommons](https://github.com/mlcommons/training_results_v0.5/tree/master/v0.5.0/nvidia/submission/code/recommendation/pytorch)

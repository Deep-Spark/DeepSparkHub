# DLRM

## Model Description

DLRM (Deep Learning Recommendation Model) is a state-of-the-art neural network architecture designed specifically for
recommendation systems. It effectively handles both categorical and numerical features, making it ideal for
personalization tasks. DLRM employs a unique architecture that combines embedding tables for categorical data with
fully-connected layers for numerical features. Its specialized parallelization scheme uses model parallelism for
embedding tables and data parallelism for dense layers, optimizing memory usage and computational efficiency. DLRM
serves as a benchmark for recommendation system development and performance evaluation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

```sh
# Prepare PaddleRec
git clone -b master --recursive https://github.com/PaddlePaddle/PaddleRec.git
cd PaddleRec
git checkout eb869a15b7d858f9f3788d9b25af4f61a022f9c4

# Prepare Criteo dataset
pushd datasets/criteo
sh run.sh
popd
```

### Install Dependencies

```sh
pip3 install -r requirements.txt

```

## Model Training

```sh
cd models/rank/dlrm
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml
python3 -u ../../../tools/infer.py -m config_bigdata.yaml

```

## Model Results

| Model | GPUs       | IPS | AUC      |
|-------|------------|-----|----------|
| DLRM  | BI-V100 x1 | 300 | 0.802409 |

## References

- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec/tree/release/2.3.0/models/rank/dlrm)

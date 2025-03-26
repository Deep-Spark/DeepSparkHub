# Wide&Deep

## Model Description

Wide&Deep is a hybrid recommendation model developed by Google that combines the strengths of memorization and
generalization. It integrates a wide linear model for capturing specific feature interactions with a deep neural network
for learning complex patterns. This architecture effectively balances precise memorization of historical data with the
ability to generalize to unseen combinations. Wide&Deep has proven particularly effective in large-scale recommendation
systems, offering improved performance in tasks like app recommendation while maintaining computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.3.0     |  22.12  |

## Model Preparation

### Prepare Resources

```sh
# Prepare PaddleRec
git clone -b release/2.3.0 https://github.com/PaddlePaddle/PaddleRec.git

# Download dataset
pushd PaddleRec/datasets/criteo/
sh run.sh
popd
```

### Install Dependencies

```sh
cd PaddleRec/
pip3 install -r requirements.txt
```

## Model Training

```sh
cd models/rank/wide_deep
export FLAGS_cudnn_exhaustive_search=True
export FLAGS_cudnn_batchnorm_spatial_persistent=True
export CUDA_VISIBLE_DEVICES=0

# Training
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml

# Evaluation
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## References

- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec)

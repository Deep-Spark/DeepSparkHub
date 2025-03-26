# xDeepFM

## Model Description

xDeepFM is an advanced deep learning model for recommendation systems that combines explicit and implicit feature
interactions. It introduces the Compressed Interaction Network (CIN) to explicitly learn high-order feature
combinations, addressing limitations of traditional factorization machines. xDeepFM integrates CIN with deep neural
networks, enabling both explicit and implicit feature learning. This architecture significantly reduces manual feature
engineering while improving recommendation accuracy. Particularly effective for sparse data, xDeepFM excels in tasks
like click-through rate prediction, offering enhanced performance in large-scale recommendation scenarios.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

```sh
# Prepare PaddleRec
git clone -b release/2.3.0 https://github.com/PaddlePaddle/PaddleRec.git

# Download dataset
pushd datasets/criteo/
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
cd models/rank/xdeepfm

# Training
python3 -u ../../../tools/trainer.py -m config_bigdata.yaml

# Evaluation
python3 -u ../../../tools/infer.py -m config_bigdata.yaml
```

## Model Results

| Model   | GPUs       | IPS  | AUC    |
|---------|------------|------|--------|
| xDeepFM | BI-V100 x1 | 6000 | 0.7955 |

## References

- [PaddleRec](https://github.com/PaddlePaddle/PaddleRec/tree/master/models/rank/xdeepfm)

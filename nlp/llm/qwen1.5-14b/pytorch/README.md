# Qwen1.5-14B (Firefly)

## Model Description

Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of
data. In comparison with the previous released Qwen, the improvements include:8 model sizes, including 0.5B, 1.8B, 4B,
7B, 14B, 32B and 72B dense models, and an MoE model of 14B with 2.7B activated;Significant performance improvement in
Chat models;Multilingual support of both base and chat models;Stable support of 32K context length for models of all
sizes;No need of trust_remote_code.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.09  |

## Model Preparation

### Prepare Resources

```sh
mkdir -p checkpoint
mv /path/to/Qwen1.5-14B checkpoint/

# download school_math_0.25M.jsonl
mkdir -p data
mv /path/to/school_math_0.25M.jsonl data/
```

### Install Dependencies

```sh
# install firefly
pushd <deepsparkhub_root>/toolbox/firefly
python3 setup.py develop
pip install transformers-stream-generator
popd
```

## Model Training

```sh
bash train.sh 16 configs/qwen-14b-sft-full.json full  
```

## Model Results

| No. | model       | peft     | num_gpus | train_samples_per_second |
|-----|-------------|----------|----------|--------------------------|
| 1   | Qwen1.5-14B | Full sft | 16       | 2.099                    |

## References

- [Firefly](https://github.com/yangjianxin1/Firefly)

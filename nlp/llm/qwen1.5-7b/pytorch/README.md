# Qwen1.5-7B (Firefly)

## Model description

Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language model pretrained on a large amount of
data. In comparison with the previous released Qwen, the improvements include:8 model sizes, including 0.5B, 1.8B, 4B,
7B, 14B, 32B and 72B dense models, and an MoE model of 14B with 2.7B activated;Significant performance improvement in
Chat models;Multilingual support of both base and chat models;Stable support of 32K context length for models of all
sizes;No need of trust_remote_code.

## Step 1: Installation

```sh
# install firefly
pushd <deepsparkhub_root>/toolbox/firefly
python3 setup.py develop
pip install transformers-stream-generator
popd
```

## Step 2: Preparing datasets and checkpoints

```sh
mkdir -p checkpoint
mv /path/to/Qwen1.5-7B checkpoint/

# download school_math_0.25M.jsonl
mkdir -p data
mv /path/to/school_math_0.25M.jsonl data/
```

## Step 3: Training

```sh
# how to train

# train with Lora
bash train.sh 1 configs/qwen-7b-sft-lora.json lora

# train with sft full
bash train.sh 16 configs/qwen-7b-sft-full.json full
```

## Results

| No. | model      | peft     | num_gpus | train_samples_per_second |
|-----|------------|----------|----------|--------------------------|
| 1   | Qwen1.5-7B | Full sft | 16       | 9.805                    |

## Reference

- [Firefly](https://github.com/yangjianxin1/Firefly)

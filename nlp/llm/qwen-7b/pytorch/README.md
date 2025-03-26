# Qwen-7B (Firefly)

## Model Description

Qwen-7B is the 7B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba
Cloud. Qwen-7B is a Transformer-based large language model, which is pretrained on a large volume of data, including web
texts, books, codes, etc. Additionally, based on the pretrained Qwen-7B, we release Qwen-7B-Chat, a large-model-based AI
assistant, which is trained with alignment techniques.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 3.4.0     |  24.06  |

## Model Preparation

### Prepare Resources

```sh
mkdir -p checkpoint
mv /path/to/Qwen-7B checkpoint/

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
# train with Lora
bash train.sh 1 configs/qwen-7b-sft-lora.json lora

# train with Ptuning-V2
bash train.sh 1 configs/qwen-7b-sft-ptuning_v2.json ptuning_v2

# train with sft full
bash train.sh 16 configs/qwen-7b-sft-full.json full
```

## Model Results

| Model   | peft       | num_gpus | train_samples_per_second |
|---------|------------|----------|--------------------------|
| Qwen-7B | Full sft   | 16       | 12.430                   |
| Qwen-7B | LoRA       | 1        | 3.409                    |
| Qwen-7B | Ptuning_V2 | 1        | 4.827                    |

## References

- [Firefly](https://github.com/yangjianxin1/Firefly)

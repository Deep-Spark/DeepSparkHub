# Bloom-7B1 (Firefly)

## Model Description

BLOOM is an autoregressive Large Language Model (LLM), trained to continue text from a prompt on vast amounts of text
data using industrial-scale computational resources. As such, it is able to output coherent text in 46 languages and 13
programming languages that is hardly distinguishable from text written by humans. BLOOM can also be instructed to
perform text tasks it hasn't been explicitly trained for, by casting them as text generation tasks.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 3.4.0     |  24.06  |

## Model Preparation

### Prepare Resources

```sh
# you can download dataset from huggingface, website here: https://huggingface.co/datasets/BelleGroup/school_math_0.25M
mkdir -p data && cd data

# you can download weights from hugginface, website here: https://huggingface.co/bigscience/bloom-7b1
mkdir -p checkpoint && cd checkpoint
```

### Install Dependencies

```sh
# install firefly
pushd <deepsparkhub_root>/toolbox/firefly
pip install transformers==4.37.2
python3 setup.py develop
popd
```

## Model Training

```sh
# train with lora
bash train.sh 1 configs/bloom-sft-lora.json lora

# train with sft full 
bash train.sh 16 configs/bloom-sft-full.json full
```

## Model Results

| Model     | GPU     | peft     | num_gpus | train_samples_per_second | train_steps_per_second |
|-----------|---------|----------|----------|--------------------------|------------------------|
| Bloom-7B1 | BI-V150 | QLoRA    | 1        | 2.041                    | 0.128                  |
| Bloom-7B1 | BI-V150 | Full sft | 16       | 4.587                    | 0.072                  |

## References

- [Firefly](https://github.com/yangjianxin1/Firefly)

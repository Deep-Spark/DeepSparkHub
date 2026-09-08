# Qwen2.5-7B(SWIFT)

## Model Description

Qwen2.5 is an advanced large language model series developed by Alibaba Cloud, offering significant improvements over
its predecessor. With enhanced capabilities in coding, mathematics, and structured data processing, it supports context
lengths up to 128K tokens and generates outputs up to 8K tokens. The model excels in multilingual support across 29
languages and demonstrates robust performance in instruction following and role-play scenarios. Qwen2.5's optimized
architecture and specialized expert models make it a versatile tool for diverse AI applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 5.0.0    |  26.09  |

## Model Preparation

### Prepare Resources

```sh
git clone https://github.com/modelscope/ms-swift -b v4.5.2
cd ms-swift

mkdir -p Qwen
# download https://huggingface.co/Qwen/Qwen2.5-7B into Qwen folder
# download https://huggingface.co/Qwen/Qwen2.5-7B-Instruct into Qwen folder
```

### Install Dependencies

```sh
# install ms-swift
pip install -e .
```

## Model Training

### Pre-training
```bash
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift pt \
    --model Qwen/Qwen2.5-7B \
    --dataset swift/chinese-c4 \
    --streaming true \
    --tuner_type full \
    --deepspeed zero2 \
    --output_dir output \
    --max_steps 10000
```

### Fine-tuning
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-en \
    --tuner_type lora \
    --output_dir output
```

### RLHF
```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --tuner_type lora \
    --output_dir output
```

## References

- [ms-swift](https://github.com/modelscope/ms-swift)

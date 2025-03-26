# Qwen2.5-7B SFT (LLaMA-Factory)

## Model Description

Qwen2.5 is an advanced large language model series developed by Alibaba Cloud, offering significant improvements over
its predecessor. With enhanced capabilities in coding, mathematics, and structured data processing, it supports context
lengths up to 128K tokens and generates outputs up to 8K tokens. The model excels in multilingual support across 29
languages and demonstrates robust performance in instruction following and role-play scenarios. Qwen2.5's optimized
architecture and specialized expert models make it a versatile tool for diverse AI applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources

```sh
# get qwen2.7-7b from https://huggingface.co/Qwen/Qwen2.5-7B and put it in checkpoints/Qwen2.5-7B
mkdir -p checkpoints
```

### Install Dependencies

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory/
git checkout d8a5571be7fcdc6f9e2442a832252d507f58c862
cp ../qwen2_5-7b_full_sft.yaml examples/train_full/
cp ../qwen2_5-7b_lora_sft.yaml examples/train_lora/
pip3 install -r requirements.txt
```

## Model Training

```sh
# Full SFT
llamafactory-cli train examples/train_full/qwen2_5-7b_full_sft.yaml

# LoRA SFT
llamafactory-cli train examples/train_lora/qwen2_5-7b_lora_sft.yaml
```

## Model Results

| Model      | GPUs        | type | train_samples_per_second |
|------------|-------------|------|--------------------------|
| Qwen2.5-7b | BI-V150 x 8 | full | 1.889                    |

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

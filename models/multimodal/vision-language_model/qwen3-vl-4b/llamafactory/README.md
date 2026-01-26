# Qwen3-VL-4B (LLaMA-Factory)

## Model Description

Meet Qwen3-VL — the most powerful vision-language model in the Qwen series to date.

This generation delivers comprehensive upgrades across the board: superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.

Available in Dense and MoE architectures that scale from edge to cloud, with Instruct and reasoning‑enhanced Thinking editions for flexible, on‑demand deployment.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.4.0  |  26.03  |

## Model Preparation

### Prepare Resources

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory/
git checkout v0.9.4
mkdir -p Qwen
# download https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct into Qwen folder
mkdir -p llamafactory
# dowload https://huggingface.co/datasets/llamafactory/RLHF-V into llamafactory folder
```

### Install Dependencies

```sh
# install llamafactory
pip install -e .
```

## Model Training

```bash
# full sft
llamafactory-cli train examples/train_full/qwen3vl_full_sft.yaml
# sft
llamafactory-cli train examples/train_lora/qwen3vl_lora_sft.yaml
# dpo
llamafactory-cli train examples/train_lora/qwen3vl_lora_dpo.yaml
```

## Model Results

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git)


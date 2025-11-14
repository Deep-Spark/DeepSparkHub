# Llama3-8B (LLaMA-Factory)

## Model Description

Llama3-8B is an advanced auto-regressive language model developed by Meta, featuring 8 billion parameters. It utilizes
an optimized transformer architecture with Grouped-Query Attention (GQA) for improved inference efficiency. Trained on
sequences of 8,192 tokens and using a 128K token vocabulary, it excels in various natural language tasks. The model
incorporates supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human
preferences, ensuring both helpfulness and safety in its responses. Llama3-8B offers state-of-the-art performance in
language understanding and generation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

```sh
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory/
git checkout 8173a88a26a1cfe78738a826047d1ef923cd4ea3
mkdir -p meta-llama
# download https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct into meta-llama folder
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- transformers-4.45.2+corex.4.3.0-py3-none-any.whl
- accelerate-0.34.2+corex.4.3.0-py3-none-any.whl

```sh
pip install llamafactory==0.9.2
pip install peft==0.11.1
```

## Model Training

```sh
# please set val_size with 0.01 in yaml to disable eval
# dpo
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
# kto
llamafactory-cli train examples/train_lora/llama3_lora_kto.yaml
# pretrain
llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml
# reward
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml
# sft
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
# full sft
pip install transformers==4.49.0
llamafactory-cli train examples/train_full/llama3_full_sft.yaml
```

## Model Results

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git)

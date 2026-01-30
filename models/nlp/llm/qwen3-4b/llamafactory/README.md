# Qwen3-4B (LLaMA-Factory)

## Model Description

Qwen3-4B-Instruct-2507 has the following features:

- Type: Causal Language Models
- Training Stage: Pretraining & Post-training
- Number of Parameters: 4.0B
- Number of Paramaters (Non-Embedding): 3.6B
- Number of Layers: 36
- Number of Attention Heads (GQA): 32 for Q and 8 for KV
- Context Length: 262,144 natively.

NOTE: This model supports only non-thinking mode and does not generate <think></think> blocks in its output. Meanwhile, specifying enable_thinking=False is no longer required.

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
# download https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 into Qwen folder
```

### Install Dependencies

```sh
# install llamafactory
pip install -e .
```

## Model Training

```bash
# full sft
llamafactory-cli train examples/train_full/qwen3_full_sft.yaml
# sft
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
# dpo
llamafactory-cli train examples/train_lora/qwen3_lora_dpo.yaml
# kto
llamafactory-cli train examples/train_lora/qwen3_lora_kto.yaml
# pretrain
llamafactory-cli train examples/train_lora/qwen3_lora_pretrain.yaml
# reward
llamafactory-cli train examples/train_lora/qwen3_lora_reward.yaml
```

## Model Results

## References

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory.git)


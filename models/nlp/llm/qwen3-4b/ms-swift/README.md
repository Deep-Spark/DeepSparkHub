# Qwen3-4B (SWIFT)

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
| BI-V150 | 5.0.0  |  26.09  |

## Model Preparation

### Prepare Resources

```sh
git clone https://github.com/modelscope/ms-swift -b v4.5.2
cd ms-swift

mkdir -p Qwen
# download https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507 into Qwen folder
# download https://huggingface.co/Qwen/Qwen3-4B-Base into Qwen folder
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
    --model Qwen/Qwen3-4B-Base \
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
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset AI-ModelScope/alpaca-gpt4-data-en \
    --tuner_type lora \
    --output_dir output
```

### RLHF
```bash
CUDA_VISIBLE_DEVICES=0 swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
    --tuner_type lora \
    --output_dir output
```

## References

- [ms-swift](https://github.com/modelscope/ms-swift)
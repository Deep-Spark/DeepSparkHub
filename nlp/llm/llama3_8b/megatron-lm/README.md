# Llama3-8B (Megatron-LM)

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
| BI-V150 | 4.3.0     |  25.09  |

## Model Preparation

### Prepare Resources

```sh
mkdir -p dataset
pushd dataset
# get gpt_small_117M_llama3.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M_llama3.tar
tar -xf gpt_small_117M_llama3.tar
rm -f gpt_small_117M_llama3.tar
popd

mkdir -p llama3-8b
# get LLM-Research/Meta-Llama-3-8B tokenizer.json put into llama3-8b
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:
- transformers-4.45.2+corex.4.3.0-py3-none-any.whl

## Model Training

```sh
bash llama3_8b_dp2_pp8_tp1.sh
```

## Model Results

## References

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

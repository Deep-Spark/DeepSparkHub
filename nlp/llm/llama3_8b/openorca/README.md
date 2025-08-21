# Llama3-8B (OpenRLHF)

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
| BI-V150 | 4.3.0     |  25.06  |

## Model Preparation

### Install OpenRLHF

```sh
# install
git clone https://github.com/OpenRLHF/OpenRLHF.git -b v0.5.7
cd OpenRLHF
pip install -e .
```

## Model Training

```sh
# Make sure you have need 16 BI-V150
cp *.sh OpenRLHF/examples/scripts/
cd OpenRLHF/examples/scripts/

# train sft
bash train_sft_llama.sh
```

tips: 如果执行中遇到oom，可以适当降低下micro_train_batch_size

## References

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)

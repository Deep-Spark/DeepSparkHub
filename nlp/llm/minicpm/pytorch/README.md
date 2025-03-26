# MiniCPM (DeepSpeed)

## Model Description

MiniCPM is a series of on-device large language models, with the core language model, MiniCPM-2B, possessing 2.4 billion
non-embedding parameters. On comprehensive benchmarks, it performs similarly to Mistral-7B (with superior capabilities
in Chinese, mathematics, and code), while exhibiting overall performance surpassing models like Llama2-13B, MPT-30B, and
Falcon-40B. Furthermore, on the MT-Bench, currently the closest benchmark to user experience, MiniCPM-2B outperforms
many representative open-source large language models, including Llama2-70B-Chat, Vicuna-33B, Mistral-7B-Instruct-v0.1,
and Zephyr-7B-alpha.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |

## Model Preparation

### Install Dependencies

```bash
cd /model/to/minicpm/pytorch
pip3 install -r requirements.txt
cd finetune
pip3 install -r requirements.txt
```

## Model Training

```bash
# SFT
bash sft_finetune.sh

# LoRA
bash lora_finetune.sh
```

## References

- [MiniCPM](https://github.com/OpenBMB/MiniCPM/tree/main)

# Phi-3

## Model Description

Phi-3 is a family of efficient small language models (SLMs) developed by Microsoft, designed to deliver high performance
while maintaining cost-effectiveness. These models excel in various tasks including language understanding, reasoning,
coding, and mathematical problem-solving. Despite their compact size, Phi-3 models outperform larger models in their
class, offering a balance between computational efficiency and capability. Their open-source nature and optimized
architecture make them ideal for applications requiring lightweight yet powerful language processing solutions across
diverse domains.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |

## Model Preparation

### Install Dependencies

```bash
pip3 install -r requirements.txt
```

## Model Training

```bash
# LoRA
bash run_lora.sh

# QLoRA
bash run_qlora.sh
```

## References

- [Phi-3](https://github.com/microsoft/Phi-3CookBook/commit/b899f6f26bcf0a140eb0e814373458740ead02c3)

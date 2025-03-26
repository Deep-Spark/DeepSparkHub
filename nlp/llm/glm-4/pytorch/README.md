# GLM-4

## Model Description

GLM-4-9B is a cutting-edge open-source language model from Zhipu AI's GLM-4 series. With 9 billion parameters, it excels
in diverse tasks including semantics, mathematics, and code generation. Its chat variant, GLM-4-9B-Chat, offers advanced
features like web browsing, code execution, and long-context inference up to 128K tokens. The model supports 26
languages and demonstrates exceptional performance in multilingual scenarios. Designed for both research and practical
applications, GLM-4-9B represents a significant advancement in large language model technology with its enhanced
capabilities and versatility.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |

## Model Preparation

### Install Dependencies

```bash
cd /model/to/GLM-4/finetune_demo
pip3 install -r requirements.txt
```

## Model Training

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune.py  data/  THUDM/glm-4-9b-chat  configs/lora.yaml
```

## References

- [GLM-4](https://github.com/THUDM/GLM-4/tree/main)

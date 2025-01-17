# GLM-4

## Model description

GLM-4-9B is an open-source version from the latest generation of pre-trained models, the GLM-4 series, released by Zhipu AI. In evaluations across datasets encompassing semantics, mathematics, reasoning, code generation, and knowledge, both GLM-4-9B and its human preference-aligned variant, GLM-4-9B-Chat, demonstrate high performance. Beyond multi-turn conversations, GLM-4-9B-Chat is equipped with advanced features like web browsing, code execution, custom function calls, and long-text inference (supporting a maximum of 128K context). This generation of models enhances multilingual support, encompassing 26 languages including Japanese, Korean, and German. We have also launched models supporting 1M context length (approximately 2 million Chinese characters).

## Step 1: Installation
```bash
cd /model/to/GLM-4/finetune_demo
pip3 install -r requirements.txt
```

## Step 2 : Training
```bash
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8  finetune.py  data/  THUDM/glm-4-9b-chat  configs/lora.yaml
```

## Reference

- [GLM-4](https://github.com/THUDM/GLM-4/tree/main)


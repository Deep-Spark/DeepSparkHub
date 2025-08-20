# Qwen3-8B grpo (verl)

## Model Description

Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support, with the following key features:

- Uniquely support of seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue) within single model, ensuring optimal performance across various scenarios.
- Significantly enhancement in its reasoning capabilities, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
- Superior human preference alignment, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
- Expertise in agent capabilities, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
- Support of 100+ languages and dialects with strong capabilities for multilingual instruction following and translation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.4.0     |  25.09  |

## Environment Preparation

### Install Dependencies
```bash
git clone https://github.com/volcengine/verl.git -b v0.5.0
cd verl
cp -rf toolbox/verl/v0.5.0/patches/* ./
pip3 install -r requirements.txt
python3 setup.py install

pip install qwen_vl_utils transformers==4.52.0
```

### Prepare Resources

```bash
python3 examples/data_preprocess/gsm8k.py
mkdir -p /home/datasets/verl/
mv ~/data/gsm8k /home/datasets/verl/gsm8k

mkdir -p /home/model_zoos/verl/
# download Qwen3-8B and put to /home/model_zoos/verl/Qwen3-8B
```

## Model Training

### train on geo3k
```bash
cd nlp/llm/qwen3-8b/verl
bash run_qwen3-8b_gsm8k.sh
```

## Model Results

## References

- [verl](https://github.com/volcengine/verl/tree/v0.5.0)

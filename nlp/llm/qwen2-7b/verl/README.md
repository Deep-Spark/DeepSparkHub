# Qwen2.5-7B grpo (verl)

## Model Description

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics, reasoning, etc. Qwen2-7B-Instruct supports a context length of up to 131,072 tokens, enabling the processing of extensive inputs.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.07  |

## Environment Preparation

### Install Dependencies
```bash
cd toolbox/verl/v0.5.0
pip3 install -r requirements.txt
python3 setup.py install
```

### Prepare Resources

```bash
python3 examples/data_preprocess/gsm8k.py
mv ~/data/gsm8k /home/datasets/verl/gsm8k

# download Qwen2.5-7B-Instruct and put to /home/model_zoos/verl/Qwen2.5-7B-Instruct

```

## Model Training

### train on gsm8k
```bash
cd nlp/llm/qwen2-7b/verl
bash run_qwen2_7B_gsm8k.sh
```

## Model Results

## References

- [verl](https://github.com/volcengine/verl/tree/v0.5.0)

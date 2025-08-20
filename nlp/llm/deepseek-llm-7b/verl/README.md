# deepseek-llm-7b-chat ppo (verl)

## Model Description

Introducing DeepSeek LLM, an advanced language model comprising 7 billion parameters. It has been trained from scratch on a vast dataset of 2 trillion tokens in both English and Chinese. In order to foster research, we have made DeepSeek LLM 7B/67B Base and DeepSeek LLM 7B/67B Chat open source for the research community.

deepseek-llm-7b-chat is a 7B parameter model initialized from deepseek-llm-7b-base and fine-tuned on extra instruction data.

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
```

### Prepare Resources

```bash
python3 examples/data_preprocess/gsm8k.py
mkdir -p /home/datasets/verl/
mv ~/data/gsm8k /home/datasets/verl/gsm8k

mkdir -p /home/model_zoos/verl/
# download deepseek-ai/deepseek-llm-7b-chat and put to /home/model_zoos/verl/deepseek-llm-7b-chat
```

## Model Training

### train on gsm8k
```bash
cd nlp/llm/deepseek-llm-7b/verl
bash run_deepseek7b_llm_gsm8k.sh
```

## Model Results

## References

- [verl](https://github.com/volcengine/verl/tree/v0.5.0)

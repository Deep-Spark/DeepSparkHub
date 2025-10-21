# gemma-2-2b-it ppo (verl)

## Model Description

Gemma is a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models. They are text-to-text, decoder-only large language models, available in English, with open weights for both pre-trained variants and instruction-tuned variants. Gemma models are well-suited for a variety of text generation tasks, including question answering, summarization, and reasoning. Their relatively small size makes it possible to deploy them in environments with limited resources such as a laptop, desktop or your own cloud infrastructure, democratizing access to state of the art AI models and helping foster innovation for everyone.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | dev-only     |  25.09  |

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
# download google/gemma-2-2b-it and put to /home/model_zoos/verl/gemma-2-2b-it
```

## Model Training

### train on gsm8k
```bash
cd nlp/llm/gemma-2-2b-it/verl
bash run_gemma_gsm8k.sh
```

## Model Results

## References

- [verl](https://github.com/volcengine/verl/tree/v0.5.0)

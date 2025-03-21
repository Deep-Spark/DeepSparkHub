# ChatGLM-6B (DeepSpeed)

## Model description

ChatGLM-6B is an open bilingual language model based on [General Language Model (GLM)](https://github.com/THUDM/GLM)
framework, with 6.2 billion parameters. With the quantization technique, users can deploy locally on consumer-grade
graphics cards (only 6GB of GPU memory is required at the INT4 quantization level).

ChatGLM-6B uses technology similar to ChatGPT, optimized for Chinese QA and dialogue. The model is trained for about 1T
tokens of Chinese and English corpus, supplemented by supervised fine-tuning, feedback bootstrap, and reinforcement
learning wit human feedback. With only about 6.2 billion parameters, the model is able to generate answers that are in
line with human preference.

## Step 1: Installation

```sh
# Install requirements
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```sh
# Get AdvertiseGen.tar.gz
wget -O AdvertiseGen.tar.gz https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1
tar xf AdvertiseGen.tar.gz
# Get chatglm-6b from https://huggingface.co/THUDM/chatglm-6b.
```

## Step 3: Training

If you load the model locally, you can change `THUDM/chatglm-6b` in `ds_train_finetune.sh` to your local model path.

```sh
cd ptuning/
bash ds_train_finetune.sh
```

## Results

| GPUs       | Toolbox   | Model      | Training speed    |
|------------|-----------|------------|-------------------|
| BI-V100 x8 | DeepSpeed | ChatGLM-6B | 0.995 samples/sec |

## Reference

- [THUDM/ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)

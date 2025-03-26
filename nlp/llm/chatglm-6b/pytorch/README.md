# ChatGLM-6B (DeepSpeed)

## Model Description

ChatGLM-6B is an open-source bilingual language model optimized for Chinese and English dialogue. With 6.2 billion
parameters, it leverages the GLM framework and advanced techniques like supervised fine-tuning and reinforcement
learning. Designed for efficient deployment, it can run on consumer-grade GPUs with INT4 quantization. ChatGLM-6B excels
in generating human-like responses, particularly in Chinese QA scenarios. Its training on extensive Chinese and English
corpora enables it to handle diverse conversational contexts while maintaining computational efficiency and
accessibility for local deployment.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

```sh
# Get AdvertiseGen.tar.gz
wget -O AdvertiseGen.tar.gz https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1
tar xf AdvertiseGen.tar.gz
# Get chatglm-6b from https://huggingface.co/THUDM/chatglm-6b.
```

### Install Dependencies

```sh
# Install requirements
pip3 install -r requirements.txt
```

## Model Training

If you load the model locally, you can change `THUDM/chatglm-6b` in `ds_train_finetune.sh` to your local model path.

```sh
cd ptuning/
bash ds_train_finetune.sh
```

## Model Results

| Model      | GPUs       | Toolbox   | Model      | Training speed    |
|------------|------------|-----------|------------|-------------------|
| ChatGLM-6B | BI-V100 x8 | DeepSpeed | ChatGLM-6B | 0.995 samples/sec |

## References

- [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)

# ChatGLM3-6B (DeepSpeed)

## Model Description

ChatGLM3-6B is the latest iteration in the ChatGLM series, developed through collaboration between Zhipu AI and Tsinghua
KEG. This open-source dialogue model builds upon its predecessors' strengths, offering enhanced conversational
capabilities and improved performance. With 6 billion parameters, it maintains a balance between computational
efficiency and language understanding. ChatGLM3-6B excels in generating coherent and contextually relevant responses,
particularly in Chinese dialogue scenarios. Its architecture supports various fine-tuning techniques, making it
adaptable for diverse applications while maintaining a low deployment threshold for practical implementation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.09  |

## Model Preparation

### Prepare Resources

```sh
# Get AdvertiseGen.tar.gz
mkdir -p data

pushd data
wget -O AdvertiseGen.tar.gz https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1
tar xf AdvertiseGen.tar.gz
popd

python3 process_data.py
```

```sh
# Get chatglm3-6b from https://modelscope.cn/models/ZhipuAI/chatglm3-6b or huggingface.
mkdir -p checkpoint

pushd checkpoint
tar -zxvf chatglm3-6b.tar.gz
popd
```

### Install Dependencies

```sh
pip3 install -r requirements.txt
```

## Model Training

```sh
bash run.sh {config_file} {num_gpus} 

# 1 GPU
bash run.sh configs/lora.yaml 1
bash run.sh configs/ptuning_v2.yaml 1

# Multi GPUs
bash run.sh configs/lora.yaml 16
bash run.sh configs/ptuning_v2.yaml 16
bash run.sh configs/sft.yaml 16
```

## Model Results

| Model      | GPU     | peft       | num_gpus | train_samples_per_second |
|------------|---------|------------|----------|--------------------------|
| ChatGLM-6B | BI-V150 | Lora       | 1        | 2.11                     |
| ChatGLM-6B | BI-V150 | ptuning_v2 | 1        | 8.889                    |
| ChatGLM-6B | BI-V150 | Lora       | 16       | 32.639                   |
| ChatGLM-6B | BI-V150 | ptuning_v2 | 16       | 115.763                  |
| ChatGLM-6B | BI-V150 | sft        | 16       | 5.99                     |

## References

- [ChatGLM3](https://github.com/THUDM/ChatGLM3)

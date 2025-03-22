# ChatGLM3-6B (DeepSpeed)

## Model description

ChatGLM3 is a generation of pre-trained dialogue models jointly released by Zhipu AI and Tsinghua KEG. ChatGLM3-6B is
the open-source model in the ChatGLM3 series, maintaining many excellent features of the first two generations such as
smooth dialogue and low deployment threshold.

## Step 1: Installation

```sh
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets and checkpoints

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

## Step 3: Training

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

## Results

| GPUs    | model      | peft       | num_gpus | train_samples_per_second |
|---------|------------|------------|----------|--------------------------|
| BI-V150 | ChatGLM-6B | Lora       | 1        | 2.11                     |
| BI-V150 | ChatGLM-6B | ptuning_v2 | 1        | 8.889                    |
| BI-V150 | ChatGLM-6B | Lora       | 16       | 32.639                   |
| BI-V150 | ChatGLM-6B | ptuning_v2 | 16       | 115.763                  |
| BI-V150 | ChatGLM-6B | sft        | 16       | 5.99                     |

## Reference

- [ChatGLM3](https://github.com/THUDM/ChatGLM3)

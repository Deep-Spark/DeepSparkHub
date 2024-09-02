# ChatGLM3-6b

## Model description

ChatGLM3 is a generation of pre-trained dialogue models jointly released by Zhipu AI and Tsinghua KEG. ChatGLM3-6B is the open-source model in the ChatGLM3 series, maintaining many excellent features of the first two generations such as smooth dialogue and low deployment threshold.

## Step 1: Installation

```bash
cd finetune_demo
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets and checkpoints

```bash
mkdir -p data && cd data
wget https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1 && mv index.html?dl=1 AdvertiseGen.tar.gz
tar -zxvf AdvertiseGen.tar.gz
cd .. && python3 process_data.py
```

```bash
mkdir -p checkpoint && cd checkpoint
```

get model chatglm3-6b from modelscope (<https://modelscope.cn/models/ZhipuAI/chatglm3-6b>) or huggingface.

```bash
tar -zxvf chatglm3-6b.tar.gz
```

## Step 3: Training

```bash
bash run.sh {config_file} {num_gpus} 

# for example
bash run.sh configs/lora.yaml 1
bash run.sh configs/ptuning_v2.yaml 1

bash run.sh configs/lora.yaml 16
bash run.sh configs/ptuning_v2.yaml 16
bash run.sh configs/sft.yaml 16
```

## Results

| No. | model      | peft      | num_gpus | train_samples_per_second |
| --- | ---------- | --------- | -------- | ------------------------ |
| 1   | ChatGLM-6b | Lora      | 1        | 2.11                     |
| 2   | ChatGLM-6b | ptuning_v2| 1        | 8.889                    |
| 3   | ChatGLM-6b | Lora      | 16       | 32.639                   |
| 4   | ChatGLM-6b | ptuning_v2| 16       | 115.763                  |
| 5   | ChatGLM-6b | sft       | 16       | 5.99                     |

## Reference

- [ChatGLM3](https://github.com/THUDM/ChatGLM3)

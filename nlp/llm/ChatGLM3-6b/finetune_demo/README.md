# ChatGLM3-6b

## Model description

    ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的对话预训练模型。ChatGLM3-6B 是 ChatGLM3 系列中的开源模型，在保留了前两代模型对话流畅、部署门槛低等众多优秀特性的基础上，ChatGLM3-6B 引入了如下特性：
    更强大的基础模型： ChatGLM3-6B 的基础模型 ChatGLM3-6B-Base 采用了更多样的训练数据、更充分的训练步数和更合理的训练策略。在语义、数学、推理、代码、知识等不同角度的数据集上测评显示， ChatGLM3-6B-Base 具有在 10B 以下的基础模型中最强的性能。
    更完整的功能支持： ChatGLM3-6B 采用了全新设计的 Prompt 格式 ，除正常的多轮对话外。同时原生支持工具调用（Function Call）、代码执行（Code Interpreter）和 Agent 任务等复杂场景。
    更全面的开源序列： 除了对话模型 ChatGLM3-6B 外，还开源了基础模型 ChatGLM3-6B-Base 、长文本对话模型 ChatGLM3-6B-32K 和进一步强化了对于长文本理解能力的 ChatGLM3-6B-128K。以上所有权重对学术研究完全开放，在填写问卷进行登记后亦允许免费商业使用。

## Step 1: Installation

```bash
$ cd finetune_demo
$ pip3 install -r requirements.txt
```

## Step 2: Preparing datasets and checkpoints

```bash
$ mkdir -p data && cd data
$ wget https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1 && mv index.html?dl=1 AdvertiseGen.tar.gz
$ tar -zxvf AdvertiseGen.tar.gz
$ cd .. && python3 process_data.py
```

```bash
$ mkdir -p checkpoint && cd checkpoint
```
get model chatglm3-6b from modelscope (https://modelscope.cn/models/ZhipuAI/chatglm3-6b) or huggingface.

```bash
$ tar -zxvf chatglm3-6b.tar.gz
```

## Step 3: Training

```bash
$ bash run.sh {config_file} {num_gpus} 

# for example
$ bash run.sh configs/lora.yaml 1
$ bash run.sh configs/ptuning_v2.yaml 1

$ bash run.sh configs/lora.yaml 16
$ bash run.sh configs/ptuning_v2.yaml 16
$ bash run.sh configs/sft.yaml 16
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


# Colossal-AI LLaMA-7B

## Model description
LLaMA-7B is part of a collection of foundation language models called **LLaMA**, which range from 7B to 65B parameters. LLaMA models are trained on trillions of tokens from publicly available datasets, and achieve state-of-the-art performance on various natural language understanding tasks. LLaMA-7B is the smallest model in the LLaMA family, but it still has impressive capabilities. It can generate fluent and coherent text, answer
questions, complete sentences, and more.

ColossalChat is the project to implement LLM with RLHF, powered by the Colossal-AI project.

Coati stands for ColossalAI Talking Intelligence. It is the name for the module implemented in this project and is also the name of the large language model developed by the ColossalChat project.

## Step 1: Installation

### Install ColossalAI
LLaMA-7B model is using ColossalAI toolbox. Before you run this model, you need to setup ColossalAI first.

```shell
cd ../../../../toolbox/ColossalAI/
bash install_toolbox_colossalai.sh
```

### Install coati
```shell
cd ColossalAI/applications/Chat
pip3 install .
```
## Step 2: Preparing datasets

You can download dataset and pretrained mode from below link.
- instinwild_en.json: [BaiduPan](https://pan.baidu.com/s/1f22_1dcWr-ZwErOo8OwbzQ?pwd=x3s9), [GoogleDrive](https://drive.google.com/file/d/1qOfrl0RIWgH2_b1rYCEVxjHF3u3Dwqay/view)
- [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)

## Step 3: Training
```shell
# multi node
torchrun --nnodes=$NODE_NUMS --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=8 --master_port $MASTER_PORT examples/train_sft.py \
    --pretrain /path/to/llama-7b-hf \
    --model 'llama' \
    --strategy colossalai_zero2_cpu \
    --log_interval 10 \
    --save_path  ${MODEL_SAVE_PATH} \
    --dataset /path/to/instinwild_en.json \
    --batch_size 1 \
    --accumulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1
```
 If the torchrun command cannot be found，you can execute:
```shell
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/
```
## Results
| Model       | Training speed     |
|-------------|-----------------|
| LLaMA-7B    | 0.9 samples/sec |

## Reference

- [ColossalAI](https://github.com/hpcaitech/ColossalAI)

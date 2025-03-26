# LLaMA-7B (ColossalAI)

## Model Description

LLaMA-7B is a foundational language model developed by Meta AI, part of the LLaMA family ranging from 7B to 65B
parameters. Trained on trillions of tokens from publicly available datasets, it demonstrates state-of-the-art
performance in natural language understanding tasks. Despite being the smallest in its family, LLaMA-7B excels in text
generation, question answering, and sentence completion. Its efficient architecture enables impressive capabilities
while maintaining computational feasibility, making it a versatile tool for various NLP applications and research in
language model development.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.1.0     |  23.09  |

## Model Preparation

### Prepare Resources

You can download dataset and pretrained mode from below link.

- instinwild_en.json: [BaiduPan](https://pan.baidu.com/s/1f22_1dcWr-ZwErOo8OwbzQ?pwd=x3s9),
  [GoogleDrive](https://drive.google.com/file/d/1qOfrl0RIWgH2_b1rYCEVxjHF3u3Dwqay/view)
- [llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)

### Install Dependencies

#### Install ColossalAI

LLaMA-7B model is using ColossalAI toolbox. Before you run this model, you need to setup ColossalAI first.

```sh
cd ../../../../toolbox/ColossalAI/v0.3.0/
bash install_toolbox_colossalai.sh
```

#### Install coati

```sh
cd ColossalAI/applications/Chat
pip3 install .
```

## Model Training

```sh
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

```sh
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/
```

## Model Results

| Model    | GPU     | Training speed  |
|----------|---------|-----------------|
| LLaMA-7B | BI-V100 | 0.9 samples/sec |

## References

- [ColossalAI](https://github.com/hpcaitech/ColossalAI)

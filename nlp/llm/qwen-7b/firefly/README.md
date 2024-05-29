# Qwen-7B

## Model description

Qwen-7B is the 7B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen-7B is a Transformer-based large language model, which is pretrained on a large volume of data, including web texts, books, codes, etc. Additionally, based on the pretrained Qwen-7B, we release Qwen-7B-Chat, a large-model-based AI assistant, which is trained with alignment techniques.

## Step 1: Installation

```bash
# install firefly
pushd <deepsparkhub_root>/toolbox/firefly
pip3 install -r requirements.txt
python3 setup.py develop
popd
```

## Step 2: Preparing datasets and checkpoints

```bash
pip install modelscope
python3 ./get_Qwen-7B.py
mkdir -p /home/model_zoo/nlp
mv /root/.cache/modelscope/hub/qwen/Qwen-7B /home/model_zoo/nlp
```

## Step 3: Training

```bash
# how to train

# train with sft full
bash train.sh 16 configs/qwen-7b-sft-full.json full

# train with Lora
bash train.sh 1 configs/qwen-7b-sft-lora.json lora

# train with Ptuning-V2
bash train.sh 1 configs/qwen-7b-sft-ptuning_v2.json ptuning_v2
```

## Results

| No.  | model     | peft        |    num_gpus        |train_samples_per_second |
| ---- | --------- | ----------- | ------------------ | ----------------------  |
| 1    | qwn-7B | Full sft    | 16                 |         12.430          |
| 2    | qwn-7B | LoRA        | 1                  |         3.409         |
| 3    | qwn-7B | Ptuning_V2  | 1                  |         4.827         |

## Reference

- [Firefly](https://github.com/yangjianxin1/Firefly)

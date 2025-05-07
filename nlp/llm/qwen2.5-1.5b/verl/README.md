# Qwen2.5-1.5B grpo (verl)

## Model Description

Qwen2.5 is an advanced large language model series developed by Alibaba Cloud, offering significant improvements over
its predecessor. With enhanced capabilities in coding, mathematics, and structured data processing, it supports context
lengths up to 128K tokens and generates outputs up to 8K tokens. The model excels in multilingual support across 29
languages and demonstrates robust performance in instruction following and role-play scenarios. Qwen2.5's optimized
architecture and specialized expert models make it a versatile tool for diverse AI applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.06  |

## Model Preparation

### Prepare Resources

```bash
# gsm8k
python3 data_preprocess/gsm8k.py --local_dir data/gsm8k
# knights-and-knaves
python3 data_preprocess/kk.py --local_dir data/kk --data_path path_to_kk/train/people3_num1000.jsonl  # please download knights-and-knaves dataset on huggingface

# download Qwen2.5-1.5B-Instruct and place at checkpoints/
```

### Install Dependencies
```bash
cd deepsparkhub/toolbox/verl
pip3 install -r requirements.txt
python3 setup.py install
```

## Model Training

wandb is strongly recommended, so please register wandb and get your wandb token, then
```bash
export WANDB_API_KEY=YOUR_WANDB_TOKEN
```
otherwise, please set trainer.logger=["console"] in config file

Tips: if throw AttributeError : module 'ixformer.inference.functions' has no attribute 'copy_blocks', pls change `ops.copy_blocks` to `ops.vllm_copy_blocks`.

### train on gsm8k
```bash
bash run_qwen2_5_1_5B_gsm8k.sh
```
### train on kk
```bash
bash run_qwen2_5_1_5B_kk.sh
```

## Model Results

## References

- [verl](https://github.com/volcengine/verl/tree/0dc8e8596b11416cddf457a02adf7009f1a10265)

# Llama3-8B (Megatron-DeepSpeed)

## Model description

Llama 3 is an auto-regressive language model that uses an optimized transformer architecture. Llama 3 uses a tokenizer with a vocabulary of 128K tokens, and was trained on on sequences of 8,192 tokens. Grouped-Query Attention (GQA) is used for all models to improve inference efficiency. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

## Step 1: Installation

```bash
# Clone
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed

# Install
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
pip3 install urllib3==1.23
```

## Step 2: Preparing datasets

```bash
pushd dataset
# get gpt_small_117M_llama3.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M_llama3.tar
tar -xf gpt_small_117M_llama3.tar
rm -f gpt_small_117M_llama3.tar
popd
```

## Step 3: Training

```bash
export NCCL_SOCKET_IFNAME="eth0"
cd examples/llama3
bash run_te_llama3_8b_node1.sh
```

## Results

|  GPUs   |             Model              | Training speed |
| :-----: | :----------------------------: | :------------: |
| BI-V150 | Llama3-8B (Megatron-DeepSpeed) |                |

## Reference

- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

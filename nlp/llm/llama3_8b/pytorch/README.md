# Llama3-8B (Megatron-DeepSpeed)

## Model Description

Llama3-8B is an advanced auto-regressive language model developed by Meta, featuring 8 billion parameters. It utilizes
an optimized transformer architecture with Grouped-Query Attention (GQA) for improved inference efficiency. Trained on
sequences of 8,192 tokens and using a 128K token vocabulary, it excels in various natural language tasks. The model
incorporates supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human
preferences, ensuring both helpfulness and safety in its responses. Llama3-8B offers state-of-the-art performance in
language understanding and generation.

## Model Preparation

### Prepare Resources

```sh
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed

pushd dataset
# get gpt_small_117M_llama3.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M_llama3.tar
tar -xf gpt_small_117M_llama3.tar
rm -f gpt_small_117M_llama3.tar
popd
```

### Install Dependencies

```sh
# install
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
```

## Model Training

```sh
export NCCL_SOCKET_IFNAME="eth0"
cd examples/llama3
bash run_te_llama3_8b_node1.sh
```

## References

- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

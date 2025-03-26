# Llama2-7B RLHF (Megatron-DeepSpeed)

## Model Description

Llama2-7B RLHF is an advanced language model developed by Meta, enhanced through Reinforcement Learning with Human
Feedback (RLHF). With 7 billion parameters, it combines the foundational capabilities of Llama2 with improved alignment
to human preferences. The RLHF process refines the model's responses, making them more coherent, contextually relevant,
and aligned with ethical guidelines. This model excels in various natural language tasks, offering enhanced performance
in dialogue systems, content generation, and instruction following while maintaining computational efficiency compared
to larger models.

**Notion: You would better to fine-tune this two models, then do RLHF training as below. So that can get good training result.**

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 3.4.0     |  24.06  |

## Model Preparation

### Prepare Resources

Download dataset and convert it.

```sh
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed/

pushd dataset/
# get gpt_small_117M.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M.tar
tar -xf gpt_small_117M.tar
rm -f gpt_small_117M.tar
popd

# Download checkpoints as above and put them to proper path, then convert checkpoints.
pushd checkpoints
bash download_rlhf_checkpoints.sh
bash convert_hf_2_meg.sh
popd
```

## Model Training

```sh
cd examples/llama2
# Modify run_llama2_7b_rlhf_node1.sh according to your machine: for example, HOST_NAME, ADDR_ARRAY, CONTAINER_NAME, NCCL_SOCKET_IFNAME, DATA_PATH
bash run_llama2_7b_rlhf_node1.sh
```

## References

- [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [Tiny-llama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b)
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

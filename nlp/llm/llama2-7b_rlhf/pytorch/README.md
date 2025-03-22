# Llama2-7B RLHF (Megatron-DeepSpeed)

In this example, we use [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b) and
[Tiny-llama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b) to do RLHF training. You
can get them in huggingface through links provided.

**Notion: You would better to fine-tune this two models, then do RLHF training as below. So that can get good training result.**

## Step 1: Install

```sh
# install
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
```

## Step 2: Dataset

Download dataset and convert it.

```sh
cd dataset
# get gpt_small_117M.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M.tar
tar -xf gpt_small_117M.tar
rm -f gpt_small_117M.tar
```

## Step 3: Train

```sh
cd examples/llama2
# Modify run_llama2_7b_1node.sh according to your machine: for example, HOST_NAME, ADDR_ARRAY, CONTAINER_NAME, NCCL_SOCKET_IFNAME
bash run_llama2_7b_rlhf_node1.sh
```

## Reference

- [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b)
- [Tiny-llama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-240k-503b)
- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

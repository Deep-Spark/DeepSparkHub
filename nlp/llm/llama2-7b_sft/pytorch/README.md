# Llama-2-7B SFT (Megatron-DeepSpeed)

## Model description

Llama 2 is a large language model released by Meta in 2023, with parameters ranging from 7B to 70B. Compared to LLaMA,
the training corpus of Llama 2 is 40% longer, and the context length has been upgraded from 2048 to 4096, allowing for
understanding and generating longer texts.

## Step 1: Installation

```sh
# install
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed
bash build_megatron-deepspeed.sh && bash install_megatron-deepspeed.sh
```

## Step 2: Preparing datasets

```sh
cd dataset
# get gpt_small_117M.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M.tar
tar -xf gpt_small_117M.tar
rm -f gpt_small_117M.tar
```

## Step 3: Training

```sh
cd examples/llama2
# Modify run_llama2_7b_1node.sh according to your machine: for example, HOST_NAME, ADDR_ARRAY, CONTAINER_NAME, NCCL_SOCKET_IFNAME
bash run_meg_llama2_7b_node1.sh
```

If the torchrun command cannot be found，you can execute:

```sh
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/
```

## Results

| GPUs       | Toolbox            | Model         | Training speed    |
|------------|--------------------|---------------|-------------------|
| BI-V100 x8 | Megatron-DeepSpeed | LLaMA2-7B SFT | 1.146 samples/sec |

## Reference

- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)

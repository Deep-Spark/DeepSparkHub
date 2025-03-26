# Llama-2-7B SFT (Megatron-DeepSpeed)

## Model Description

Llama-2-7B SFT is a fine-tuned version of Meta's Llama-2-7B model, optimized using supervised fine-tuning techniques.
With 7 billion parameters and an extended 4096-token context window, it excels in understanding and generating coherent,
contextually relevant text. The SFT process enhances its performance on specific tasks by leveraging labeled datasets,
making it particularly effective for applications requiring precise language understanding and generation. This model
combines the foundational capabilities of Llama-2 with task-specific optimizations, offering improved performance while
maintaining computational efficiency.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.1     |  24.03  |

## Model Preparation

### Prepare Resources

```sh
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed

pushd dataset
# get gpt_small_117M.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M.tar
tar -xf gpt_small_117M.tar
rm -f gpt_small_117M.tar
popd
```

## Model Training

```sh
cd examples/llama2
# Modify run_llama2_7b_1node.sh according to your machine: for example, HOST_NAME, ADDR_ARRAY, CONTAINER_NAME, NCCL_SOCKET_IFNAME
bash run_meg_llama2_7b_node1.sh
```

If the torchrun command cannot be found，you can execute:

```sh
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/
```

## Model Results

| Model         | GPUs       | Toolbox            | Training speed    |
|---------------|------------|--------------------|-------------------|
| LLaMA2-7B SFT | BI-V100 x8 | Megatron-DeepSpeed | 1.146 samples/sec |

## References

- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

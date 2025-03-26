# Llama-2-7B (Megatron-DeepSpeed)

## Model Description

Llama-2-7B is an advanced large language model developed by Meta, offering improved capabilities over its predecessor.
With 7 billion parameters, it's trained on 40% more data and supports a doubled context length of 4096 tokens, enabling
better understanding of longer texts. This model excels in various natural language tasks, including text generation and
comprehension. Its enhanced architecture and training methodology make it a powerful tool for AI applications while
maintaining computational efficiency compared to larger models in the Llama-2 series.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V100 | 3.1.0     |  23.12  |

## Model Preparation

### Prepare Resources

```sh
cd <DeepSparkHub_Root>/toolbox/Megatron-DeepSpeed

cd dataset
# get gpt_small_117M.tar
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/gpt_small_117M.tar
tar -xf gpt_small_117M.tar
rm -f gpt_small_117M.tar
```

## Model Training

```sh
cd examples/llama2
# Modify run_llama2_7b_1node.sh according to your machine: for example, HOST_NAME, ADDR_ARRAY, CONTAINER_NAME, NCCL_SOCKET_IFNAME
bash run_llama2_7b_1node.sh
```

If the torchrun command cannot be found，you can execute:

```sh
ln -s /usr/local/corex-3.1.0/lib64/python3/dist-packages/bin/torchrun /usr/local/bin/
```

## Model Results

| Model      | GPUs       | Toolbox            | Training speed    |
|------------|------------|--------------------|-------------------|
| Llama-2-7B | BI-V100 x8 | Megatron-DeepSpeed | 1.263 samples/sec |

## References

- [Megatron-DeepSpeed](https://github.com/microsoft/Megatron-DeepSpeed)

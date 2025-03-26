# DeepSeekMoE 7B (ColossalAI)

## Model Description

DeepSeekMoE 7B is an efficient variant of the DeepSeekMoE 16B model, utilizing a Mixture-of-Experts (MoE) architecture.
This model features innovative strategies like fine-grained expert segmentation and shared experts isolation, enabling
it to handle complex language tasks effectively. With 7 billion parameters, it balances computational efficiency and
performance, making it suitable for various natural language processing applications. DeepSeekMoE 7B demonstrates strong
capabilities in language understanding and generation while maintaining a more compact structure compared to its larger
counterpart.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources

Get "deepseek-moe-16b-base" models and config file from huggingface or other place, and mv it to
"ColossalAI/examples/language/deepseek/deepseek-ai/deepseek-moe-16b-base". One recommended link:
"<https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/tree/main>".

```sh
git clone -b v0.4.4 https://github.com/hpcaitech/ColossalAI.git --depth=1
cd ColossalAI/

pushd examples/language/deepseek
mkdir -p deepseek-ai
mv <Path>/deepseek-moe-16b-base deepseek-ai/
popd
```

### Install Dependencies

Firstly, you should ensure that ColossalAI is installed in the environment. Generally, ColossalAI is installed by
default.

```sh
cp -rf <DeepSparkHub_Root>/toolbox/ColossalAI/v0.4.4/patches/* ./
pip3 install .
```

## Model Training

```sh
cd ColossalAI/examples/language/deepseek
colossalai run --nproc_per_node 16 benchmark.py -c 7b -g  -b 16 --tp 1 --pp 4 --num_steps 50
```

## Model Results

| Model          | GPU     | Training speed   |
|----------------|---------|------------------|
| DeepSeekMoE 7B | BI-V150 | 6.85 samples/sec |

## References

- [ColossalAI](https://github.com/hpcaitech/ColossalAI/tree/v0.4.4/examples/language/deepseek)

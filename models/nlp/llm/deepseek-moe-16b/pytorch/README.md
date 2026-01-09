# DeepSeekMoE 16B (ColossalAI)

## Model Description

DeepSeekMoE 16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It employs an innovative MoE architecture, which involves two principal strategies: fine-grained expert segmentation and shared experts isolation. It is trained from scratch on 2T English and Chinese tokens, and exhibits comparable performance with DeekSeek 7B and LLaMA2 7B, with only about 40% of computations. For research purposes, we release the model checkpoints of DeepSeekMoE 16B Base and DeepSeekMoE 16B Chat to the public, which can be deployed on a single GPU with 40GB of memory without the need for quantization

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.2.0     |  25.03  |
| BI-V150 | 4.1.1     |  24.12  |

## Model Preparation

### Prepare Resources

Get "deepseek-moe-16b-base" models and config file from huggingface or other place, and mv it to
"ColossalAI/examples/language/deepseek/deepseek-ai/deepseek-moe-16b-base". One recommended link:
"<https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/tree/main>".

```sh
git clone -b v0.4.8 https://github.com/hpcaitech/ColossalAI.git --depth=1
cd ColossalAI/

pushd examples/language/deepseek
mkdir -p deepseek-ai
mv <Path>/deepseek-moe-16b-base deepseek-ai/
popd
```

### Install Dependencies

Contact the Iluvatar administrator to get the missing packages:

- bitsandbytes-*.whl

```sh
pip install bitsandbytes-*.whl
cp -rf <DeepSparkHub_Root>/toolbox/ColossalAI/v0.4.8/patches/* ./
pip3 install .
```

## Model Training

```sh
cd ColossalAI/examples/language/deepseek
colossalai run --nproc_per_node 16 benchmark.py -c 7b -g -b 16 --tp 1 --pp 4 --num_steps 50
```

## Model Results

| Model          | GPU     | Training speed   |
|----------------|---------|------------------|
| DeepSeekMoE 16B | BI-V150 | 6.85 samples/sec |

## References

- [ColossalAI](https://github.com/hpcaitech/ColossalAI/tree/v0.4.8/examples/language/deepseek)

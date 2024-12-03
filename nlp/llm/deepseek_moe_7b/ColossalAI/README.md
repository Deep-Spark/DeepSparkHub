# Colossal-AI LLaMA-7B

## Model description
DeepSeekMoE 16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It employs an innovative MoE architecture, which involves two principal strategies: fine-grained expert segmentation and shared experts isolation.
DeepSeekMoE 7B is a variant of the 16B model. 

## Step 1: Install

Firstly, you should ensure that ColossalAI is installed in the environment. Generally, ColossalAI is installed by default.

## Step 2: Prepare model and config

Get "deepseek-moe-16b-base" models and config file from huggingface or other place, and mv it to "/home/model_zoos/nlp/deepseek-moe-16b-base".
One recommended link: "https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/tree/main".

## Step 3: Training
```bash
$ bash deepseek_7b_pretrain.sh
```

## Results
| Model              | Training speed     |
|--------------------|--------------------|
| deepseek-moe-7b    |  6.85 samples/sec  |

## Reference

- [ColossalAI](https://github.com/hpcaitech/ColossalAI/tree/v0.4.4/examples/language/deepseek)

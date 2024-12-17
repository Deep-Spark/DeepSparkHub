# DeepSeekMoE 7B (ColossalAI)

## Model description

DeepSeekMoE 7B is a variant of the 16B model.

DeepSeekMoE 16B is a Mixture-of-Experts (MoE) language model with 16.4B parameters. It employs an innovative MoE architecture, which involves two principal strategies: fine-grained expert segmentation and shared experts isolation.

## Step 1: Install

Firstly, you should ensure that ColossalAI is installed in the environment. Generally, ColossalAI is installed by default.

```sh
git clone -b v0.4.4 https://github.com/hpcaitech/ColossalAI.git --depth=1
cd ColossalAI
cp -rf <DeepSparkHub_Root>/toolbox/ColossalAI/v0.4.4/patches/* ./
pip3 install . 
```

## Step 2: Prepare model and config

Get "deepseek-moe-16b-base" models and config file from huggingface or other place, and mv it to "ColossalAI/examples/language/deepseek/deepseek-ai/deepseek-moe-16b-base".
One recommended link: "<https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/tree/main>".

```bash
cd ColossalAI/examples/language/deepseek
mkdir -p deepseek-ai
mv <Path>/deepseek-moe-16b-base deepseek-ai/
```

## Step 3: Training

```bash
cd ColossalAI/examples/language/deepseek
colossalai run --nproc_per_node 16 benchmark.py -c 7b -g  -b 16 --tp 1 --pp 4 --num_steps 50
```

## Results

| Model              | Training speed     |
|--------------------|--------------------|
| deepseek-moe-7b    |  6.85 samples/sec  |

## Reference

- [ColossalAI](https://github.com/hpcaitech/ColossalAI/tree/v0.4.4/examples/language/deepseek)

# Llama3-8B SFT (ColossalAI)

## Model description

The Llama 3 Herd of models natively supports multilinguality, coding, reasoning, and tool usage. Our largest model is dense Transformer with 405B parameters, processing information in a context window of up to 128K tokens, Llama 3 8B is the smallest model of Llama 3 Herd of models.

## Step 1: Preparing checkpoints

Get "Meta-Llama-3-8B" models and config file from modelscope or other place, and mv it to "/home/model_zoos/".
One recommended link: "<https://modelscope.cn/models/LLM-Research/Meta-Llama-3-8B>".

```sh
mkdir -p /home/model_zoos/
mv <Path>/Meta-Llama-3-8B /home/model_zoos/

wget http://files.deepspark.org.cn:880/deepspark/data/tokenizer/tokenizer.model
cp tokenizer.model /home/model_zoos/Meta-Llama-3-8B
```

## Step 2: Installation and preparing datasets

You should ensure that the corresponding version of ColossalAI has been installed in the iluvatar environment. Then install applications as follows:

```sh
git clone -b v0.4.4 https://github.com/hpcaitech/ColossalAI.git --depth=1
cd ColossalAI
cp -rf <DeepSparkHub_Root>/toolbox/ColossalAI/v0.4.4/patches/* ./
cd applications/Colossal-LLaMA
pip3 install -e . 

# preparing datasets
wget http://files.deepspark.org.cn:880/deepspark/data/datasets/school_math_0.25M.jsonl
mkdir -p dataset/school_math/convert/
mv school_math_0.25M.jsonl dataset/school_math
bash ./prepare_sft_dataset.sh llama3
```

## Step 3: Training

```sh
bash run_llama3_8b_sft_3d.sh
```

## Results

| model     | peft        |    num_gpus        |train_samples_per_second |
| --------- | ----------- | ------------------ | ----------------------  |
| llama3-8b | Full sft    | 16                 |         1.53            |

## Reference

- [ColossalAI (tag:v0.4.4)](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA)

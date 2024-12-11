# Llama3-8B SFT (ColossalAI)

## Model description

The Llama 3 Herd of models natively supports multilinguality, coding, reasoning, and tool usage. Our largest model is dense Transformer with 405B parameters, processing information in a context window of up to 128K tokens, Llama 3 8B is the smallest model of Llama 3 Herd of models.

## Step 1: Installation

Firstly, you should ensure that the corresponding version of ColossalAI has been installed in the iluvatar environment. Then install applications as follows:

```sh
cd ColossalAI/applications/Colossal-LLaMA
pip3 install -e . 
```

## Step 2: Preparing datasets and checkpoints

```sh
pip3 install modelscope
python3 ./get_Meta_LLaMA_8B.py
mkdir -p /home/model_zoos/nlp
mv ~/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B /home/model_zoos/nlp

wget http://files.deepspark.org.cn:880/deepspark/tokenizer.model
cp tokenizer.model /home/model_zoos/nlp/Meta-Llama-3-8B

wget http://files.deepspark.org.cn:880/deepspark/school_math_0.25M.jsonl
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

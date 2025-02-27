# Yi-6B

## Model description

The Yi series models are the next generation of open-source large language models trained from scratch by 01.AI. Targeted as a bilingual language model and trained on 3T multilingual corpus, the Yi series models become one of the strongest LLM worldwide, showing promise in language understanding, commonsense reasoning, reading comprehension, and more. 

## Step 1: Installation

```sh
pip3 install -r requirements.txt
```

## Step 2: Preparing checkpoints

```sh
mkdir -p /home/model_zoos/nlp/Yi-6B

pip3 install modelscope
modelscope download --model 01ai/Yi-6B --local_dir /home/model_zoos/nlp/Yi-6B

```

## Step 3: Training

```sh
bash scripts/run_sft_Yi_6b.sh
```

## Results

| No. | model      | peft     | num_gpus | train_samples_per_second |
|-----|------------|----------|----------|--------------------------|
| 1   | Yi-6B      | Full sft | 16       | 0.11                     |

## Reference

- [Yi](https://github.com/01-ai/Yi/tree/main?tab=readme-ov-file)

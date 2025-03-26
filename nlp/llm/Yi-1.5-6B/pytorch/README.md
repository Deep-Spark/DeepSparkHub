# Yi-1.5-6B (DeepSpeed)

## Model Description

The Yi series models are the next generation of open-source large language models trained from scratch by 01.AI.
Targeted as a bilingual language model and trained on 3T multilingual corpus, the Yi series models become one of the
strongest LLM worldwide, showing promise in language understanding, commonsense reasoning, reading comprehension, and
more.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V150 | 4.2.0     |  25.03  |

## Model Preparation

### Prepare Resources

```sh
mkdir -p /home/model_zoos/nlp/Yi-1.5-6B

pip3 install modelscope
modelscope download --model 01ai/Yi-1.5-6B --local_dir /home/model_zoos/nlp/Yi-1.5-6B
```

### Install Dependencies

```sh
pip3 install -r requirements.txt
```

## Model Training

```sh
bash scripts/run_sft_Yi-1.5-6b.sh
```

## Model Results

| Model     | GPU     | peft     | num_gpus | train_samples_per_second |
|-----------|---------|----------|----------|--------------------------|
| Yi-1.5-6B | BI-V150 | Full sft | 16       | 0.11                     |

## References

- [Yi](https://github.com/01-ai/Yi/tree/main?tab=readme-ov-file)

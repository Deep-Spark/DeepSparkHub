# Bart

## Model description
BART is sequence-to-sequence model trained with denoising as pretraining objective. We show that this pretraining objective is more generic and show that we can match RoBERTa results on SQuAD and GLUE and gain state-of-the-art results on summarization (XSum, CNN dataset), long form generative question answering (ELI5) and dialog response genration (ConvAI2). 

## Envoronment
Before you run this model, please refer to [README_environment.md](README_environment.md) to setup Fairseq and install requirements.

## Download data

```
chmod -R 777 fairseq
cd fairseq
mkdir -p glue_data

cd glue_data
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip

unzip RTE.zip
rm -rf RTE.zip
```

## Preprocess data

```
cd ..
./examples/roberta/preprocess_GLUE_tasks.sh glue_data RTE
```

## Download pretrain weight

```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```

## Finetune on CLUE RTE task

```
bash bart.sh
```

## Inference on GLUE RTE task

```
python3 bart.py
```

## Results on BI-V100

```
| GPUs | QPS | Train Epochs | Accuracy  |
|------|-----|--------------|------|
| 1x8  | 113.18 | 10           | 83.8 |
```
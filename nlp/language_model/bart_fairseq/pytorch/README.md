# BART

## Model description
BART is sequence-to-sequence model trained with denoising as pretraining
objective. We show that this pretraining objective is more generic and
show that we can match RoBERTa results on SQuAD and GLUE and gain 
state-of-the-art results on summarization (XSum, CNN dataset), 
long form generative question answering (ELI5) and dialog response
genration (ConvAI2). 

## Step 1: Installation
Bart model is using Fairseq toolbox. Before you run this model, 
you need to setup Fairseq first.

```bash
# Go to "toolbox/Fairseq" directory in root path
cd ../../../../toolbox/Fairseq/
bash install_toolbox_fairseq.sh
```

## Step 2: Preparing datasets

```bash
# Download dataset
cd fairseq/
mkdir -p glue_data
cd glue_data/
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip
unzip RTE.zip
rm -rf RTE.zip

# Preprocess dataset
cd ..
./examples/roberta/preprocess_GLUE_tasks.sh glue_data RTE

# Download pretrain weight
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzvf bart.large.tar.gz
```

## Step 3: Training
```bash
# Finetune on CLUE RTE task
bash bart.sh

# Inference on GLUE RTE task
`
python3 bart.py
```

## Results

| GPUs | QPS | Train Epochs | Accuracy  |
|------|-----|--------------|------|
| BI-v100 x8  | 113.18 | 10           | 83.8 |

## Reference
- [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)

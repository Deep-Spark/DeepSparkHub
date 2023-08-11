# RoBERTa

## Model description
RoBERTa iterates on BERT's pretraining procedure, including training the model 
longer, with bigger batches over more data; removing the next sentence prediction
objective; training on longer sequences; and dynamically changing the masking
pattern applied to the training data.

## Step 1: Installation
RoBERTa model is using Fairseq toolbox. Before you run this model, 
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
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz
```

## Step 3: Training

```bash
# Finetune on CLUE RTE task
bash roberta.sh

# Inference on GLUE RTE task
python3 roberta.py
```

## Results

| GPUs | QPS | Train Epochs | Accuracy  |
|------|-----|--------------|------|
| BI-v100 x8  | 207.5 | 10           | 86.3 |

## Reference
- [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)

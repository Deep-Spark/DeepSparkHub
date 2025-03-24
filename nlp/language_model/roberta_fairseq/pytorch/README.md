# RoBERTa

## Model Description

RoBERTa (Robustly optimized BERT approach) is an enhanced version of BERT that improves upon the original model's
pretraining methodology. It removes the next sentence prediction objective, uses larger batches and more data, and
implements dynamic masking patterns. These optimizations allow RoBERTa to achieve better performance across various NLP
tasks. By training on longer sequences and optimizing the training procedure, RoBERTa demonstrates superior language
understanding capabilities compared to its predecessor, making it a powerful tool for natural language processing
applications.

## Model Preparation

### Prepare Resources

```bash
# Go to "toolbox/Fairseq" directory in root path
cd ../../../../toolbox/Fairseq/

# Download dataset
cd fairseq/
mkdir -p glue_data
cd glue_data/
wget https://dl.fbaipublicfiles.com/glue/data/RTE.zip
unzip RTE.zip
rm -rf RTE.zip

# Preprocess dataset
cd ../
./examples/roberta/preprocess_GLUE_tasks.sh glue_data RTE

# Download pretrain weight
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz
```

### Install Dependencies

RoBERTa model is using Fairseq toolbox. Before you run this model, you need to setup Fairseq first.

```bash
bash install_toolbox_fairseq.sh
```

## Model Training

```bash
# Finetune on CLUE RTE task
bash roberta.sh

# Inference on GLUE RTE task
python3 roberta.py
```

## Model Results

| Model   | GPUs       | QPS   | Train Epochs | Accuracy |
|---------|------------|-------|--------------|----------|
| RoBERTa | BI-v100 x8 | 207.5 | 10           | 86.3     |

## References

- [Fairseq](https://github.com/facebookresearch/fairseq/tree/v0.10.2)

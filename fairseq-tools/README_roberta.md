# Roberta

## Model description
RoBERTa iterates on BERT's pretraining procedure, including training the model longer, with bigger batches over more data; removing the next sentence prediction objective; training on longer sequences; and dynamically changing the masking pattern applied to the training data.

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
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz
```

## Finetune on CLUE RTE task

```
bash roberta.sh
```

## Inference on GLUE RTE task

```
python3 roberta.py
```

## Results on BI-V100

```
| GPUs | QPS | Train Epochs | Accuracy  |
|------|-----|--------------|------|
| 1x8  | 207.5 | 10           | 86.3 |
```
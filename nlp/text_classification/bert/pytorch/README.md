# Text Classification

# Bert-base WNLI 

## Model description

Bert-base WNLI task Fine-tuning

## Step 1: Installing packages

``` shell
cd  <your_project_path>/nlp/text_classification/bert/pytorch
pip3 install -r requirements.txt
```

## Step 2: Training

### On single GPU

``` shell
bash train.sh
```

### Multiple GPUs on one machine

```shell
bash train_dist.sh
```
## Results on BI-V100

| GPUs | Samples/s | Loss |
|------|-----------|------|
| 1x1  | 144.5     | 0.74 |
| 1x8  | 322.74    | 0.71 |

## Reference
https://github.com/huggingface/

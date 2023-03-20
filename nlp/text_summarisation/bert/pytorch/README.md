# Bert-base summarization 

## Model description

Bert-base summarization task Fine-tuning

## Step 1: Installing packages

``` shell
cd  <your_project_path>/nlp/text_summarisation/bert/pytorch
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

| GPUs | Samples/s | Loss   |
|------|-----------|--------|
| 1x1  | 16.71  | 1.8038 |
| 1x8  | 117.576  | 1.8288 |

## Reference
https://github.com/huggingface/

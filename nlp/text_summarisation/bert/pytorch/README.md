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
| 1x1  | 1834.099  | 0.0281 |
| 1x8  | 6229.625  | 0.0278 |

## Reference
https://github.com/huggingface/

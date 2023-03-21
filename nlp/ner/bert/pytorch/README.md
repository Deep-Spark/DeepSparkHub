# BERT NER

## Model description

BERT-base NER task Fine-tuning

## Step 1: Installing packages

``` shell
cd  <your_project_path>/nlp/ner/bert/pytorch
pip3 install -r requirements.txt
```

## Step 2: Training

### On single GPU

``` shell
bash run.sh
```

### Multiple GPUs on one machine

```shell
bash run_dist.sh
```
## Results on BI-V100

| GPUs | Samples/s  | Loss |
|------|------|----|
| 1x1  | 100 | 0.0696 |
| 1x8  | 252 | 0.0688 |

## Reference
https://github.com/huggingface/

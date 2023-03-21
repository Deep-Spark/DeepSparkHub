# BERT Question Answering

## Model description

BERT-base SQuAD task Fine-tuning

## Step 1: Installing packages

``` shell
cd  <your_project_path>/nlp/querstion_answering/bert/pytorch
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

| GPUs | Samples/s | F1     |
|------|-----------|--------|
| 1x1  | 128.86    | 87     |
| 1x8  | 208.6     | 78.69  |

## Reference
https://github.com/huggingface/

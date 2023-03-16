# Bert-base squad 

## Model description

Bert-base squad task Fine-tuning

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

| GPUs | Samples/s | Loss   |
|------|-----------|--------|
| 1x1  | 29.86     | 0.9861 |
| 1x8  | 178.906   | 0.9804 |

## Reference
https://github.com/huggingface/
# T5 

## Model description

T5, or Text-to-Text Transfer Transformer, is a Transformer based architecture that uses a text-to-text approach. Every task – including translation, question answering, and classification – is cast as feeding the model text as input and training it to generate some target text. This allows for the use of the same model, loss function, hyperparameters, etc. across our diverse set of tasks. 

## Step 1: Installing packages

``` shell
cd  <your_project_path>/t5/pytorch
bash examples_ix/init_torch.sh
```

## Step 2: Training

### On single GPU

``` shell
bash examples_ix/train_t5_small_torch.sh
```

### On single GPU (AMP)

```shell
bash examples_ix/train_t5_small_amp_torch.sh
```

### Multiple GPUs on one machine

```shell
bash examples_ix/train_t5_small_dist_torch.sh
```

### Multiple GPUs on one machine (AMP)
```shell
bash examples_ix/train_t5_small_amp_dist_torch.sh
```

## Results on BI-V100

| GUSs | Samples/s  | Loss |
|------|------|----|
| 1x1  | 339 | 1.18 |
| 1x8  | 2488 | 1.18 |

## Reference
https://github.com/huggingface/
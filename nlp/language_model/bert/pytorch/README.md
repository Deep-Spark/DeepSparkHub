# BERT Pretraining

## Model description

BERT, or Bidirectional Encoder Representations from Transformers, improves upon standard Transformers by removing the unidirectionality constraint by using a masked language model (MLM) pre-training objective. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective enables the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, BERT uses a next sentence prediction task that jointly pre-trains text-pair representations.


## Step 1: Installing

```shell
bash init.sh
```

## Step 2: Preparing dataset

Reference: [training_results_v1.0](https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch)

**Structure**
```
└── bert/dataset
    ├── 2048_shards_uncompressed
    ├── bert_config.json
    ├── eval_set_uncompressed
    └── model.ckpt-28252.apex.pt
```
## Step 3: Training

> Warning: The number of cards are computed by `torch.cuda.device_count()`, 
> so you can set `CUDA_VISIBLE_DEVICES` to set the number of cards.


### Multiple GPUs on one machine (AMP)

```shell
DATA=/path/to/bert/dataset bash train_bert_pretraining_amp_dist.sh
```


### Parameters
```shell
--gradient_accumulation_steps
--max_steps
--train_batch_size
--eval_batch_size 
--learning_rate
--target_mlm_accuracy
--dist_backend
```


## Results on BI-V100

| GPUs | FP16 | FPS |  E2E   | MLM Accuracy |
| ---- | ---- | --- | ------ | ------------ |
| 1x8  | True | 227 | 13568s | 0.72         |


| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
| -------------------- | ---------------------------------------- | ----------- | -------- | ---------- | ----------- | ----------------------- | --------- |
| 0.72                 | SDK V2.2,bs:32,8x,AMP                    | 214         | 0.72     | 152\*8     | 0.96        | 20.3\*8                 | 1         |



# Reference

https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch

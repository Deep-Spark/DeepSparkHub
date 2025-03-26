# BERT Pretraining

## Model Description

BERT (Bidirectional Encoder Representations from Transformers) is a groundbreaking language model that revolutionized
natural language processing. It employs a transformer architecture with bidirectional attention, enabling it to capture
context from both directions in text. Pretrained using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
tasks, BERT achieves state-of-the-art results across various NLP tasks through fine-tuning. Its ability to understand
deep contextual relationships in text has made it a fundamental model in modern NLP research and applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 2.2.0     |  22.09  |

## Model Preparation

### Prepare Resources

Reference: [training_results_v1.0](https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch)

```bash
└── bert/dataset
    ├── 2048_shards_uncompressed
    ├── bert_config.json
    ├── eval_set_uncompressed
    └── model.ckpt-28252.apex.pt
```

### Install Dependencies

```shell
apt install -y git numactl
pip install h5py
pip install psutil
pip install mlperf-logging
pip install boto3
```

## Model Training

> Warning: The number of cards are computed by `torch.cuda.device_count()`, 
> so you can set `CUDA_VISIBLE_DEVICES` to set the number of cards.

```shell
# Multiple GPUs on one machine (AMP)
DATA=/path/to/bert/dataset bash train_bert_pretraining_amp_dist.sh

# Parameters
--gradient_accumulation_steps
--max_steps
--train_batch_size
--eval_batch_size 
--learning_rate
--target_mlm_accuracy
--dist_backend
```

## Model Results

| Model            | GPUs       | FP16 | FPS | E2E    | MLM Accuracy |
|------------------|------------|------|-----|--------|--------------|
| BERT Pretraining | BI-V100 x8 | True | 227 | 13568s | 0.72         |

| Convergence criteria | Configuration (x denotes number of GPUs) | Performance | Accuracy | Power（W） | Scalability | Memory utilization（G） | Stability |
| -------------------- | ---------------------------------------- | ----------- | -------- | ---------- | ----------- | ----------------------- | --------- |
| 0.72                 | SDK V2.2,bs:32,8x,AMP                    | 214         | 0.72     | 152\*8     | 0.96        | 20.3\*8                 | 1         |

## References

- [training_results_v1.0](https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch)

# BERT Pretraining

## Model Description

BERT (Bidirectional Encoder Representations from Transformers) is a groundbreaking language model that revolutionized
natural language processing. It employs a transformer architecture with bidirectional attention, enabling it to capture
context from both directions in text. Pretrained using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
tasks, BERT achieves state-of-the-art results across various NLP tasks through fine-tuning. Its ability to understand
deep contextual relationships in text has made it a fundamental model in modern NLP research and applications.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
| :----: | :----: | :----: |
| BI-V150 | 4.3.0     |  25.12  |

## Model Preparation

### Prepare Resources

Reference: [training_results_v1.0](https://github.com/mlcommons/training_results_v1.0/tree/master/NVIDIA/benchmarks/bert/implementations/pytorch)

```bash
mkdir -p data/datasets/bert_mini
cd data/datasets/bert_mini
wget http://files.deepspark.org.cn:880/deepspark/bert_mini/2048_shards_uncompressed_mini.tar.gz
wget http://files.deepspark.org.cn:880/deepspark/bert_mini/eval_set_uncompressed.tar.gz
wget http://files.deepspark.org.cn:880/deepspark/bert_mini/model.ckpt-28252.apex.pt
wget http://files.deepspark.org.cn:880/deepspark/bert_mini/bert_config.json
tar -xf 2048_shards_uncompressed_mini.tar.gz
tar -xf eval_set_uncompressed.tar.gz

└── data/datasets/bert_mini
    ├── 2048_shards_uncompressed
    ├── eval_set_uncompressed
    └── model.ckpt-28252.apex.pt
    └── bert_config.json
```

### Install Dependencies

```shell
apt install -y numactl

cd base
pip3 install -r requirements.txt
python3 setup.py install
```

## Model Training

### Training with default
```shell
bash run_training.sh \
--name default \
--config V100x1x8 \
--data_dir ../data/datasets/bert_mini/ \
--max_steps 500 \
--train_batch_size 10 \
--target_mlm_accuracy 0.33 \
--init_checkpoint "../data/datasets/bert_mini/model.ckpt-28252.apex.pt"
```

### Training with iluvatar
```shell
bash run_training.sh --name iluvatar --config 03V100x1x8 --train_batch_size 27 --data_dir ../data/datasets/bert_mini/ --master_port 22233
```

## Model Results
| Model        | GPUs       | E2E   | MLM Accuracy | training_sequences_per_second | final_loss |
|:------------:|:----------:|:-----:|:------------:|:-----------------------------:|:----------:|
| BERT default | BI-V150 x8 | 82.72s | 0.339        | 3.685                         |  4.723     |
| BERT iluvatar | BI-V150 x8 | 509.79s | 0.720       | 10513.181                     |  1.306   |

## References


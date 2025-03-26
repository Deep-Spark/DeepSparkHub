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
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

This [Google Drive location](https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) contains the
following.  
You need to download tf1_ckpt folde , vocab.txt and bert_config.json into one file named bert_pretrain_ckpt_tf

```sh
bert_pretrain_ckpt_tf: contains checkpoint files
    model.ckpt-28252.data-00000-of-00001
    model.ckpt-28252.index
    model.ckpt-28252.meta
    vocab.txt
    bert_config.json
```

[Download and preprocess datasets](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#generate-the-tfrecords-for-wiki-dataset)
You need to make a file named  bert_pretrain_tf_records and store the results above.
tips: you can git clone this repo in other place ,we need the bert_pretrain_tf_records results here.

### Install Dependencies

```shell
bash init_tf.sh
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.7.tar.gz
tar xf openmpi-4.0.7.tar.gz
cd openmpi-4.0.7/
./configure --prefix=/usr/local/bin --with-orte
make -j4 && make install
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
```

## Model Training

```shell
# Training on single card
bash run_1card_FPS.sh --input_files_dir=/path/to/bert_pretrain_tf_records/train_data \
        --init_checkpoint=/path/to/bert_pretrain_ckpt_tf/model.ckpt-28252 \
        --eval_files_dir=/path/to/bert_pretrain_tf_records/eval_data \
        --train_batch_size=6 \
        --bert_config_file=/path/to/bert_pretrain_ckpt_tf/bert_config.json

# Training on mutil-cards
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export IX_NUM_CUDA_VISIBLE_DEVICES=8
bash run_multi_card_FPS.sh --input_files_dir=/path/to/bert_pretrain_tf_records/train_data \
        --init_checkpoint=/path/to/bert_pretrain_ckpt_tf/model.ckpt-28252 \
        --eval_files_dir=/path/to/bert_pretrain_tf_records/eval_data \
        --train_batch_size=6 \
        --bert_config_file=/path/to/bert_pretrain_ckpt_tf/bert_config.json
```

## Model Results

| Model            | GPUs       | acc      | fps      |
|------------------|------------|----------|----------|
| BERT Pretraining | BI-V100 x8 | 0.424126 | 0.267241 |

# BERT

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

Download training dataset(.tf_record), eval dataset(.json), vocab.txt and checkpoint:
bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1.ckpt

```sh
cd scripts
mkdir -p squad
```

Please [BERT](https://github.com/google-research/bert#pre-training-with-bert) download vocab.txt here

- Create fine-tune dataset
  - Download dataset for fine-tuning and evaluation such as Chinese Named Entity
    Recognition[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020), Chinese sentences
    classification[TNEWS](https://github.com/CLUEbenchmark/CLUE), Chinese Named Entity
    Recognition[ChineseNER](https://github.com/zjy-ucas/ChineseNER), English question and answering[SQuAD v1.1 train
    dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [SQuAD v1.1 eval
    dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json), package of English sentences
    classification[GLUE](https://gluebenchmark.com/tasks).
  - We haven't provide the scripts to create tfrecord yet, while converting dataset files from JSON format to TFRECORD
    format, please refer to run_classifier.py or run_squad.py file in [BERT](https://github.com/google-research/bert)
    repository or the CLUE official repository
    [CLUE](https://github.com/CLUEbenchmark/CLUE/blob/master/baselines/models/bert/run_classifier.py) and
    [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version)

### Pretrained models

We have provided several kinds of pretrained checkpoint.

- [Bert-base-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_base_ascend_v130_zhwiki_official_nlp_bs256_acc91.72_recall95.06_F1score93.36/),
  trained on zh-wiki datasets with 128 length.
- [Bert-large-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_zhwiki_official_nlp_bs3072_loss0.8/),
  trained on zh-wiki datasets with 128 length.
- [Bert-large-en](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1/),
  tarined on en-wiki datasets with 512 length.

### Install Dependencies

```sh
pip3 install -r requirements.txt
```

## Model Training

```sh
bash scripts/run_squad_gpu_distribute.sh 8
```

## Model Results

| Model | GPUs       | per step time | exact_match | F1     |
|-------|------------|---------------|-------------|--------|
| BERT  | BI-V100 x8 | 1.898s        | 71.9678     | 81.422 |

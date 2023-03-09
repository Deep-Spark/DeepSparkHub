# BERT
## Model description
The BERT network was proposed by Google in 2018. The network has made a breakthrough in the field of NLP. The network uses pre-training to achieve a large network structure without modifying, and only by adding an output layer to achieve multiple text-based tasks in fine-tuning. The backbone code of BERT adopts the Encoder structure of Transformer. The attention mechanism is introduced to enable the output layer to capture high-latitude global semantic information. The pre-training uses denoising and self-encoding tasks, namely MLM(Masked Language Model) and NSP(Next Sentence Prediction). No need to label data, pre-training can be performed on massive text data, and only a small amount of data to fine-tuning downstream tasks to obtain good results. The pre-training plus fune-tuning mode created by BERT is widely adopted by subsequent NLP networks.

[Paper](https://arxiv.org/abs/1810.04805):  Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]((https://arxiv.org/abs/1810.04805)). arXiv preprint arXiv:1810.04805.

[Paper](https://arxiv.org/abs/1909.00204):  Junqiu Wei, Xiaozhe Ren, Xiaoguang Li, Wenyong Huang, Yi Liao, Yasheng Wang, Jiashu Lin, Xin Jiang, Xiao Chen, Qun Liu. [NEZHA: Neural Contextualized Representation for Chinese Language Understanding](https://arxiv.org/abs/1909.00204). arXiv preprint arXiv:1909.00204.
## Step 1: Installing
```
pip3 install -r requirements.txt
```
## Step 2: Prepare Datasets
# 1. Download training dataset(.tf_record), eval dataset(.json), vocab.txt and checkpoint：bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1.ckpt
```
cd scripts
mkdir -p squad
```
Please [BERT](https://github.com/google-research/bert#pre-training-with-bert) download vocab.txt here

- Create fine-tune dataset
    - Download dataset for fine-tuning and evaluation such as Chinese Named Entity Recognition[CLUENER](https://github.com/CLUEbenchmark/CLUENER2020), Chinese sentences classification[TNEWS](https://github.com/CLUEbenchmark/CLUE), Chinese Named Entity Recognition[ChineseNER](https://github.com/zjy-ucas/ChineseNER), English question and answering[SQuAD v1.1 train dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json), [SQuAD v1.1 eval dataset](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json), package of English sentences classification[GLUE](https://gluebenchmark.com/tasks).
    - We haven't provide the scripts to create tfrecord yet, while converting dataset files from JSON format to TFRECORD format, please refer to run_classifier.py or run_squad.py file in [BERT](https://github.com/google-research/bert) repository or the CLUE official repository [CLUE](https://github.com/CLUEbenchmark/CLUE/blob/master/baselines/models/bert/run_classifier.py) and [CLUENER](https://github.com/CLUEbenchmark/CLUENER2020/tree/master/tf_version)

# [Pretrained models](#contents)

We have provided several kinds of pretrained checkpoint.

- [Bert-base-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_base_ascend_v130_zhwiki_official_nlp_bs256_acc91.72_recall95.06_F1score93.36/), trained on zh-wiki datasets with 128 length.
- [Bert-large-zh](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_zhwiki_official_nlp_bs3072_loss0.8/), trained on zh-wiki datasets with 128 length.
- [Bert-large-en](https://download.mindspore.cn/model_zoo/r1.3/bert_large_ascend_v130_enwiki_official_nlp_bs768_loss1.1/), tarined on en-wiki datasets with 512 length.

## Step 3: Training
```
bash scripts/run_squad_gpu_distribute.sh 8
```
### [Evaluation result]
## Results on BI-V100

| GPUs | per step time  |  exact_match  |  F1  |
|------|--------------  |---------------|------|
|  1*8 |   1.898s       |   71.9678     |81.422|
### 性能数据：NV 
## Results on NV-V100s

| GPUs | per step time  |  exact_match  |  F1  |
|------|--------------  |---------------|------|
|  1*8 |   1.877s       |   71.9678     |81.422|


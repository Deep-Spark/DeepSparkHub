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
| BI-V100 | 2.3.0     |  22.12  |

## Model Preparation

### Prepare Resources

Download the [MNLI Dataset](http://www.nyu.edu/projects/bowman/multinli/)

### Install Dependencies

```bash
git clone --recursive https://github.com/PaddlePaddle/PaddleNLP.git
cd PaddleNLP
pip3 install -r requirements.txt
```

## Model Training

```bash
# Make sure your dataset path is the same as above
bash train_bert.sh
```

## Model Results

| Model            | GPU        | FP32                            |
|------------------|------------|---------------------------------|
| BERT Pretraining | BI-V100 x1 | Acc@1=84.5,FPS=5.1,BatchSize=32 |

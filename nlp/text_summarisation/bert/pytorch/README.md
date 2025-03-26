# BERT Text Summarization

## Model Description

BERT for Text Summarization is a fine-tuned version of the BERT model specifically adapted for generating concise
summaries from longer text documents. By leveraging BERT's bidirectional attention mechanism, it effectively captures
key information and contextual relationships within the text. This approach enables the model to produce coherent and
informative summaries while preserving the original meaning. BERT summarization is particularly valuable for
applications requiring efficient information extraction and condensation, such as news aggregation, document analysis,
and content curation.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Install Dependencies

``` shell
cd  <your_project_path>/nlp/text_summarisation/bert/pytorch
pip3 install -r requirements.txt
```

## Model Training

``` shell
# On single GPU
bash train.sh

# Multiple GPUs on one machine
bash train_dist.sh
```

## Model Results

| Model | GPUs       | Samples/s | Loss   |
|-------|------------|-----------|--------|
| BERT  | BI-V100 x1 | 16.71     | 1.8038 |
| BERT  | BI-V100 x8 | 117.576   | 1.8288 |

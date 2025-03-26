# BERT Question Answering

## Model Description

BERT for SQuAD (Stanford Question Answering Dataset) is a fine-tuned version of BERT specifically designed for question
answering tasks. It excels at extracting precise answers from given contexts by leveraging BERT's bidirectional
attention mechanism. The model is trained to predict the start and end positions of answers within text passages,
demonstrating exceptional performance in comprehension tasks. BERT SQuAD's ability to understand context and
relationships between words makes it particularly effective for complex question answering scenarios in various domains.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Prepare Resources

```bash
# Get "bert-base-uncased" from [Huggingface](https://huggingface.co/bert-base-uncased)

## Install lfs
wget https://packagecloud.io/github/git-lfs/packages/el/7/git-lfs-2.13.2-1.el7.x86_64.rpm/download -O lfs.rpm
rpm -ivh lfs.rpm

## Clone from Huggingface, it may take long time for large file
git lfs install
git clone https://huggingface.co/bert-base-uncased
```

### Install Dependencies

```bash
cd  <your_project_path>/nlp/querstion_answering/bert/pytorch
pip3 install -r requirements.txt
```

## Model Training

> Make sure you've got "bert-base-uncased" ready in ./bert-base-uncased

```bash
# On single GPU
## config "dataset_name" to be local path of squad if you want to run off-line
## e.g. --dataset_name /path/to/squad
bash run.sh

# Multiple GPUs on one machine
## config "dataset_name" to be local path of squad if you want to run off-line
## e.g. --dataset_name /path/to/squad
bash run_dist.sh
```

## Model Results

| Model | GPUs       | Samples/s | F1    |
|-------|------------|-----------|-------|
| BERT  | BI-V100 x1 | 128.86    | 87    |
| BERT  | BI-V100 x8 | 208.6     | 78.69 |

## References

- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

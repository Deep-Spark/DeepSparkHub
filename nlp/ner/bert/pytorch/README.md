# BERT NER

## Model Description

BERT NER (Named Entity Recognition) is a powerful application of the BERT model for identifying and classifying named
entities in text. By fine-tuning the pre-trained BERT model on NER tasks, it leverages bidirectional context to
accurately detect entities like names, locations, and organizations. This approach significantly improves entity
recognition accuracy compared to traditional methods. BERT NER's ability to understand deep contextual relationships
makes it particularly effective for complex text analysis tasks in various domains, including information extraction and
text mining.

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

``` shell
cd  <your_project_path>/nlp/ner/bert/pytorch
pip3 install -r requirements.txt
```

## Model Training

> Make sure you've got "bert-base-uncased" ready in ./bert-base-uncased.

``` shell
# On single GPU
## config "dataset_name" to be local path of conll2003 if you want to run off-line
## e.g. --dataset_name /path/to/conll2003
bash run.sh

# Multiple GPUs on one machine
## config "dataset_name" to be local path of conll2003 if you want to run off-line
## e.g. --dataset_name /path/to/conll2003
bash run_dist.sh
```

## Model Results

| Model | GPUs       | Samples/s | Loss   |
|-------|------------|-----------|--------|
| BERT  | BI-V100 x1 | 100       | 0.0696 |
| BERT  | BI-V100 x8 | 252       | 0.0688 |

## References

- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

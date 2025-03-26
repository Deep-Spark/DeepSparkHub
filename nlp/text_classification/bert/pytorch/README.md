# BERT Text Classification

## Model Description

BERT for WNLI (Winograd NLI) is a fine-tuned version of BERT specifically designed for natural language inference tasks.
It excels at determining the relationship between pairs of sentences, particularly in resolving pronoun references and
understanding context. By leveraging BERT's bidirectional attention mechanism, it can effectively capture subtle
linguistic nuances and relationships between text segments. This makes BERT WNLI particularly valuable for tasks
requiring deep comprehension of sentence structure and meaning, such as coreference resolution and textual entailment.

## Supported Environments

| GPU    | [IXUCA SDK](https://gitee.com/deep-spark/deepspark#%E5%A4%A9%E6%95%B0%E6%99%BA%E7%AE%97%E8%BD%AF%E4%BB%B6%E6%A0%88-ixuca) | Release |
|--------|-----------|---------|
| BI-V100 | 3.0.0     |  23.03  |

## Model Preparation

### Install Dependencies

``` shell
cd  <your_project_path>/nlp/text_classification/bert/pytorch
pip3 install -r requirements.txt
```

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

## Model Training

> Make sure you've got "bert-base-uncased" ready in ./bert-base-uncased

```bash
# On single GPU
bash train.sh

# Multiple GPUs on one machine
bash train_dist.sh
```

## Model Results

| Model        | GPUs       | Samples/s | Loss |
|--------------|------------|-----------|------|
| BERT WNLI FT | BI-V100 x1 | 144.5     | 0.74 |
| BERT WNLI FT | BI-V100 x8 | 322.74    | 0.71 |

## References

- [bert-base-uncased](https://huggingface.co/bert-base-uncased)

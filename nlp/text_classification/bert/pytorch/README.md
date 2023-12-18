# BERT Text Classification

## Model description

BERT-base WNLI task Fine-tuning

## Step 1: Installation

``` shell
cd  <your_project_path>/nlp/text_classification/bert/pytorch
pip3 install -r requirements.txt
```

## Step 2: Preparing datasets

```bash
# Get "bert-base-uncased" from [Huggingface](https://huggingface.co/bert-base-uncased)

## Install lfs
wget https://packagecloud.io/github/git-lfs/packages/el/7/git-lfs-2.13.2-1.el7.x86_64.rpm/download -O lfs.rpm
rpm -ivh lfs.rpm

## Clone from Huggingface, it may take long time for large file
git lfs install
git clone https://huggingface.co/bert-base-uncased
```

## Step 3: Training

**Make sure you've got "bert-base-uncased" ready in ./bert-base-uncased**

### On single GPU

```bash
bash train.sh
```

### Multiple GPUs on one machine

```bash
bash train_dist.sh
```
## Results on BI-V100

| GPUs | Samples/s | Loss |
|------|-----------|------|
| 1x1  | 144.5     | 0.74 |
| 1x8  | 322.74    | 0.71 |

## Reference
https://github.com/huggingface/
